"""CLI entrypoint for evaluation.

Evaluate a model on the wiki-search task without training.

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --model Qwen/Qwen3-4B --num_samples 20
    uv run python scripts/eval.py --model outputs/checkpoints/model --output results.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from src.environment import SandboxClient
from src.judge import get_local_judge_client
from src.trainer.episode import get_judge_reward
from src.agent import Agent, AgentConfig

# Defaults from env vars
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
DEFAULT_JUDGE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:1234/v1")
DEFAULT_TRAIN_MODEL = os.environ.get("TRAIN_MODEL", "Qwen/Qwen3-4B")


def load_model(model_path: str, device: str = "cuda"):
    """Load model from HuggingFace or local checkpoint.

    Args:
        model_path: HuggingFace model name or path to local checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_path}")

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


def load_questions(num_samples: int) -> list[dict]:
    """Load evaluation questions from dataset.

    Args:
        num_samples: Number of questions to load

    Returns:
        List of {question, answer} dicts
    """
    from datasets import load_dataset

    ds = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
    samples = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
        })
    return samples


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SLM on wiki-search")

    parser.add_argument("--model", type=str, default=DEFAULT_TRAIN_MODEL,
                        help="Model to evaluate (HF name or checkpoint path)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of eval questions (default: 10)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Max turns per question (default: 3)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens per generation (default: 1024)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Judge model (default: JUDGE_MODEL env var)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    return parser.parse_args()


async def main_async() -> None:
    """Async main function."""
    args = parse_args()

    print("=" * 60)
    print("SLM-RL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Questions: {args.num_samples}")
    print(f"Max turns: {args.max_turns}")
    print(f"Judge: {args.judge_model}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model)

    # Load questions
    print(f"\nLoading {args.num_samples} questions...")
    questions = load_questions(args.num_samples)

    # Setup judge
    judge_client = get_local_judge_client(DEFAULT_JUDGE_URL)

    # Setup sandbox
    sandbox = SandboxClient()

    # Results storage
    results = []
    correct_count = 0
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

            print(f"\nQ{i+1}: {question[:60]}...")
            print(f"   Expected: {expected}")

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

            # Record result
            result = {
                "question": question,
                "expected": expected,
                "answer": answer or "(no answer)",
                "correct": judge_result.correct,
                "num_turns": episode.num_turns,
            }
            results.append(result)

            if judge_result.correct:
                correct_count += 1
            total_turns += episode.num_turns

            # Print result
            display_answer = answer or "(no answer)"
            print(f"   Answer: {display_answer[:60]}{'...' if len(display_answer) > 60 else ''}")
            if answer is None:
                print(f"   Judge: INCORRECT (no answer)")
            elif judge_result.error:
                print(f"   Judge error: {judge_result.error}")
            else:
                print(f"   Judge: {'CORRECT' if judge_result.correct else 'INCORRECT'}")

    # Summary
    accuracy = correct_count / len(questions) * 100
    avg_turns = total_turns / len(questions)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {correct_count}/{len(questions)} ({accuracy:.1f}%)")
    print(f"Avg turns: {avg_turns:.1f}")
    print("=" * 60)

    # Save results if requested
    if args.output:
        output_data = {
            "model": args.model,
            "num_samples": args.num_samples,
            "accuracy": accuracy,
            "avg_turns": avg_turns,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
