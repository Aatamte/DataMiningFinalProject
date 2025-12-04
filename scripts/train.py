"""CLI entrypoint for training.

Start with tiny params to verify everything works:
    uv run python scripts/train.py

Then scale up:
    uv run python scripts/train.py --num_samples 100 --num_epochs 3

With live plotting:
    uv run python scripts/train.py --live-plot
"""

import argparse
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.trainer import Trainer, TrainerConfig

# Defaults from env vars
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
DEFAULT_JUDGE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:1234/v1")
DEFAULT_TRAIN_MODEL = os.environ.get("TRAIN_MODEL", "Qwen/Qwen3-4B")


def validate_environment(judge_url: str, judge_model: str, model_name: str) -> bool:
    """Validate that required services are reachable.

    Returns True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("VALIDATING ENVIRONMENT")
    print("=" * 60)

    all_ok = True

    # Check judge API is reachable
    print(f"\n[1/2] Checking judge API: {judge_url}")
    try:
        resp = httpx.get(f"{judge_url}/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json()
            model_ids = [m.get("id", "") for m in models.get("data", [])]
            print(f"  ✓ API reachable, available models: {model_ids}")

            # Check if judge model is available
            if judge_model in model_ids or any(judge_model in m for m in model_ids):
                print(f"  ✓ Judge model '{judge_model}' available")
            else:
                print(f"  ✗ Judge model '{judge_model}' not found in available models")
                print(f"    Available: {model_ids}")
                all_ok = False
        else:
            print(f"  ✗ API returned status {resp.status_code}")
            all_ok = False
    except httpx.ConnectError:
        print(f"  ✗ Cannot connect to {judge_url}")
        print(f"    Is LM Studio / Ollama running?")
        all_ok = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_ok = False

    # Check training model exists on HuggingFace (just validate format)
    print(f"\n[2/2] Checking training model: {model_name}")
    if "/" in model_name:
        print(f"  ✓ Model name format valid (will download if needed)")
    else:
        print(f"  ! Warning: Model name '{model_name}' may be local path")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - please fix before training")
    print("=" * 60 + "\n")

    return all_ok


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SLM on wiki-search")

    # Training params
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of questions to train on (default: 3)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Max tool-use turns per episode (default: 3)")
    parser.add_argument("--num_rollouts", type=int, default=2,
                        help="Rollouts per question (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max new tokens per generation (default: 1024)")

    # Model settings
    parser.add_argument("--model_name", type=str,
                        default=DEFAULT_TRAIN_MODEL,
                        help="HuggingFace model to train (default: TRAIN_MODEL env var)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model for judging (default: JUDGE_MODEL env var)")

    # Paths
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints",
                        help="Output directory for checkpoints")

    # Live plotting
    parser.add_argument("--live-plot", action="store_true",
                        help="Update plot after each step")

    # Eval mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Run episodes without training (no weight updates)")

    return parser.parse_args()


async def main_async() -> None:
    """Async main function."""
    args = parse_args()

    # Validate environment before starting
    if not validate_environment(DEFAULT_JUDGE_URL, args.judge_model, args.model_name):
        sys.exit(1)

    config = TrainerConfig(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        max_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model_name,
        judge_model=args.judge_model,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )

    trainer = Trainer(config)
    trainer.setup()
    await trainer.train(live_plot=args.live_plot)


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
