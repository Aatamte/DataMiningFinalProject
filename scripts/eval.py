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


def load_questions(q_percentage: float) -> list[dict]:
    """Load evaluation questions from dataset.

    Args:
        q_percentage: Percentage of total questions to load (0-100)

    Returns:
        List of {question, answer} dicts
    """
    from datasets import load_dataset

    ds = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
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
    parser.add_argument("--q_percentage", type=float, default=1.0,
                        help="Percentage of questions to eval (default: 1.0 = 1%%)")
    parser.add_argument("--n_per_q", type=int, default=1,
                        help="Number of eval runs per question (default: 1)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Max turns per question (default: 3)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens per generation (default: 1024)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Judge model (default: JUDGE_MODEL env var)")

    return parser.parse_args()


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

    # Load questions first to get total count
    print(f"Loading questions ({args.q_percentage}% of dataset)...")
    questions, total_questions = load_questions(args.q_percentage)

    print("=" * 60)
    print("SLM-RL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_display}")
    print(f"Mode: {'API' if USE_API else 'Local'}")
    print(f"Questions: {len(questions)} / {total_questions} ({args.q_percentage}%)")
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

    # Results storage
    results = []
    total_correct = 0
    total_runs = 0
    total_turns = 0

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    async with sandbox:
        agent_config = AgentConfig(
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
        )
        agent = Agent(model, tokenizer, sandbox, agent_config)

        for i, q in enumerate(questions):
            question = q["question"]
            expected = q["answer"]

            print(f"\nQ{i+1}/{len(questions)}: {question[:60]}...")
            print(f"   Expected: {expected}")

            q_correct = 0
            q_runs = []

            for run_idx in range(args.n_per_q):
                # Run episode
                episode = await agent.run(question)
                answer = episode.final_answer

                # Get judge result (skip if no answer)
                if answer is None:
                    from src.trainer.episode import JudgeResult
                    judge_result = JudgeResult(reward=0.0, correct=False)
                else:
                    judge_result = await get_judge_reward(
                        judge_client, args.judge_model,
                        question, expected, answer
                    )

                # Record run
                run_result = {
                    "run": run_idx + 1,
                    "answer": answer or "(no answer)",
                    "correct": judge_result.correct,
                    "num_turns": episode.num_turns,
                }
                q_runs.append(run_result)

                if judge_result.correct:
                    q_correct += 1
                    total_correct += 1
                total_runs += 1
                total_turns += episode.num_turns

                # Print run result
                if args.n_per_q > 1:
                    status = "✓" if judge_result.correct else "✗"
                    print(f"   Run {run_idx + 1}: {status}")

            # Record question result
            result = {
                "question": question,
                "expected": expected,
                "runs": q_runs,
                "accuracy": q_correct / args.n_per_q * 100,
            }
            results.append(result)

            # Print question summary
            if args.n_per_q == 1:
                display_answer = q_runs[0]["answer"]
                print(f"   Answer: {display_answer[:60]}{'...' if len(display_answer) > 60 else ''}")
                print(f"   Judge: {'CORRECT' if q_runs[0]['correct'] else 'INCORRECT'}")
            else:
                print(f"   Question accuracy: {q_correct}/{args.n_per_q} ({result['accuracy']:.0f}%)")

    # Summary
    accuracy = total_correct / total_runs * 100 if total_runs > 0 else 0
    avg_turns = total_turns / total_runs if total_runs > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total runs: {total_runs} ({len(questions)} questions × {args.n_per_q} runs)")
    print(f"Accuracy: {total_correct}/{total_runs} ({accuracy:.1f}%)")
    print(f"Avg turns: {avg_turns:.1f}")
    print("=" * 60)

    # Save results
    output_data = {
        "model": EVAL_MODEL if USE_API else args.model,
        "mode": "api" if USE_API else "local",
        "q_percentage": args.q_percentage,
        "n_per_q": args.n_per_q,
        "total_questions": total_questions,
        "num_questions": len(questions),
        "total_runs": total_runs,
        "accuracy": accuracy,
        "avg_turns": avg_turns,
        "results": results,
    }

    results_file = eval_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
