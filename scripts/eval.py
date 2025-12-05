"""CLI entrypoint for evaluation.

Evaluate a model on the wiki-search task without training.

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --model Qwen/Qwen3-4B --q_percentage 10
    uv run python scripts/eval.py --model runs/train_.../checkpoints/final --base_model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()

from src.environment import SandboxClient
from src.judge import get_local_judge_client
from src.trainer.episode import get_judge_reward
from src.agent import Agent, AgentConfig

# Defaults from env vars
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
DEFAULT_JUDGE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:1234/v1")
DEFAULT_TRAIN_MODEL = os.environ.get("TRAIN_MODEL", "Qwen/Qwen3-4B")
USE_API = os.environ.get("EVAL_USE_API", "false").lower() == "true"
EVAL_MODEL = os.environ.get("EVAL_MODEL", "")


def setup_eval_dir(model_name: str) -> Path:
    """Create timestamped eval directory.

    Args:
        model_name: Model being evaluated (used in dir name)

    Returns:
        Path to eval directory
    """
    # Clean model name for directory
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path("evals") / f"eval_{clean_name}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def is_lora_checkpoint(path: str) -> bool:
    """Check if path is a LoRA adapter checkpoint.

    Args:
        path: Path to check

    Returns:
        True if path contains adapter_config.json
    """
    adapter_config = Path(path) / "adapter_config.json"
    return adapter_config.exists()


def load_model(model_path: str, base_model: str | None = None, device: str = "cuda"):
    """Load model from HuggingFace or local checkpoint.

    Args:
        model_path: HuggingFace model name or path to local checkpoint
        base_model: Base model for LoRA checkpoints (required if model_path is LoRA)
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_path}")

    # Check if this is a LoRA checkpoint
    if is_lora_checkpoint(model_path):
        if base_model is None:
            print("ERROR: --base_model required for LoRA checkpoints")
            print(f"  Detected LoRA adapter at: {model_path}")
            print("  Specify the base model that was fine-tuned")
            sys.exit(1)

        print(f"  LoRA adapter detected, loading base model: {base_model}")

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Apply LoRA adapter
        print(f"  Applying LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("  LoRA merged into base model")
    else:
        # Regular model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Device: {device}")

    return model, tokenizer


def load_questions_for_eval(subset: str, q_percentage: float) -> tuple[list[dict], int]:
    """Load evaluation questions from dataset.

    Args:
        subset: Which subset to load - "train", "test", or "all"
        q_percentage: Percentage of subset to load (0-100)

    Returns:
        Tuple of (list of {question, answer} dicts, subset total size)
    """
    from src.environment.core import load_questions

    ds = load_questions(subset=subset)
    total = len(ds)
    num_samples = max(1, int(total * q_percentage / 100))

    samples = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
        })
    return samples, total


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SLM on wiki-search")

    parser.add_argument("--model", type=str, default=DEFAULT_TRAIN_MODEL,
                        help="Model to evaluate (HF name or LoRA checkpoint path)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for LoRA checkpoints (required if --model is LoRA)")
    parser.add_argument("--subset", type=str, default="all", choices=["train", "test", "all"],
                        help="Which data subset to evaluate: train (80%%), test (20%%), or all (default: all)")
    parser.add_argument("--q_percentage", type=float, default=100.0,
                        help="Percentage of subset to eval (default: 100.0 = 100%%)")
    parser.add_argument("--n_per_q", type=int, default=1,
                        help="Number of eval runs per question (default: 1)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Max turns per question (default: 3)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens per generation (default: 1024)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Judge model (default: JUDGE_MODEL env var)")
    parser.add_argument("--full_report", action="store_true",
                        help="Evaluate on train, test, and all subsets (overrides --subset)")

    return parser.parse_args()


async def eval_subset(
    subset: str,
    questions: list[dict],
    agent: Agent,
    judge_client,
    judge_model: str,
    n_per_q: int,
) -> dict:
    """Evaluate on a single subset.

    Returns:
        Dict with results for this subset
    """
    from src.trainer.episode import JudgeResult

    results = []
    total_correct = 0
    total_runs = 0
    total_turns = 0

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q["answer"]

        print(f"\n  Q{i+1}/{len(questions)}: {question[:55]}...")
        print(f"     Expected: {expected[:40]}...")

        q_correct = 0
        q_runs = []

        for run_idx in range(n_per_q):
            # Run episode
            episode = await agent.run(question)
            answer = episode.final_answer

            # Get judge result (skip if no answer)
            if answer is None:
                judge_result = JudgeResult(reward=0.0, correct=False, approach_score=0)
            else:
                trajectory = episode.format_for_judge()
                judge_result = await get_judge_reward(
                    judge_client, judge_model,
                    question, expected, answer,
                    trajectory=trajectory
                )

            # Record run
            run_result = {
                "run": run_idx + 1,
                "answer": answer or "(no answer)",
                "correct": judge_result.correct,
                "approach_score": judge_result.approach_score,
                "reward": judge_result.reward,
                "num_turns": episode.num_turns,
                "messages": episode.conversation.to_messages(),
            }
            q_runs.append(run_result)

            if judge_result.correct:
                q_correct += 1
                total_correct += 1
            total_runs += 1
            total_turns += episode.num_turns

            # Print run result
            if n_per_q > 1:
                status = "[OK]" if judge_result.correct else "[FAIL]"
                print(f"     Run {run_idx + 1}: {status} | Approach: {judge_result.approach_score}/100")

        # Record question result
        result = {
            "question": question,
            "expected": expected,
            "runs": q_runs,
            "accuracy": q_correct / n_per_q * 100,
        }
        results.append(result)

        # Print question summary
        if n_per_q == 1:
            run = q_runs[0]
            display_answer = run["answer"]
            print(f"     Answer: {display_answer[:55]}{'...' if len(display_answer) > 55 else ''}")
            status = "CORRECT" if run["correct"] else "INCORRECT"
            print(f"     Judge: {status} | Approach: {run['approach_score']}/100 | Reward: {run['reward']:.2f}")
        else:
            print(f"     Question accuracy: {q_correct}/{n_per_q} ({result['accuracy']:.0f}%)")

    accuracy = total_correct / total_runs * 100 if total_runs > 0 else 0
    avg_turns = total_turns / total_runs if total_runs > 0 else 0

    return {
        "subset": subset,
        "num_questions": len(questions),
        "total_runs": total_runs,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "avg_turns": avg_turns,
        "results": results,
    }


async def main_async() -> None:
    """Async main function."""
    args = parse_args()

    # Determine model name for display/logging
    if USE_API:
        model_display = f"{EVAL_MODEL} (via API)"
    else:
        model_display = args.model

    # Setup eval directory
    eval_dir = setup_eval_dir(EVAL_MODEL if USE_API else args.model)

    # Determine which subsets to evaluate
    if args.full_report:
        subsets_to_eval = ["train", "test", "all"]
    else:
        subsets_to_eval = [args.subset]

    print("=" * 60)
    print("SLM-RL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_display}")
    print(f"Mode: {'API' if USE_API else 'Local'}")
    print(f"Subsets: {', '.join(subsets_to_eval)}")
    print(f"Sample %: {args.q_percentage}%")
    print(f"Runs per question: {args.n_per_q}")
    print(f"Max turns: {args.max_turns}")
    print(f"Judge: {args.judge_model}")
    print(f"Output dir: {eval_dir}")
    print("=" * 60)

    # Load model (skip if using API)
    if USE_API:
        model, tokenizer = None, None
        print("Using API mode - skipping local model load")
    else:
        model, tokenizer = load_model(args.model, base_model=args.base_model)

    # Setup judge
    judge_client = get_local_judge_client(DEFAULT_JUDGE_URL)

    # Setup sandbox
    sandbox = SandboxClient()

    # Evaluate each subset
    all_results = {}

    async with sandbox:
        agent_config = AgentConfig(
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
        )
        agent = Agent(model, tokenizer, sandbox, agent_config)

        for subset in subsets_to_eval:
            print(f"\n{'=' * 60}")
            print(f"EVALUATING: {subset.upper()} SET")
            print("=" * 60)

            questions, total_in_subset = load_questions_for_eval(subset, args.q_percentage)
            print(f"Questions: {len(questions)} / {total_in_subset} ({args.q_percentage}%)")

            subset_results = await eval_subset(
                subset=subset,
                questions=questions,
                agent=agent,
                judge_client=judge_client,
                judge_model=args.judge_model,
                n_per_q=args.n_per_q,
            )
            subset_results["total_in_subset"] = total_in_subset
            all_results[subset] = subset_results

            print(f"\n  {subset.upper()} Accuracy: {subset_results['total_correct']}/{subset_results['total_runs']} ({subset_results['accuracy']:.1f}%)")

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for subset in subsets_to_eval:
        r = all_results[subset]
        print(f"  {subset.upper():6} : {r['total_correct']:3}/{r['total_runs']:3} ({r['accuracy']:5.1f}%) - {r['num_questions']} questions, avg {r['avg_turns']:.1f} turns")

    print("=" * 60)

    # Save results
    output_data = {
        "model": EVAL_MODEL if USE_API else args.model,
        "mode": "api" if USE_API else "local",
        "q_percentage": args.q_percentage,
        "n_per_q": args.n_per_q,
        "subsets": all_results,
    }

    results_file = eval_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Cleaning up...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    main()
