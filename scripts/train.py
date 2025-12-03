"""CLI entrypoint for training.

Start with tiny params to verify everything works:
    uv run python scripts/train.py

Then scale up:
    uv run python scripts/train.py --num_samples 100 --num_epochs 3
"""

import argparse
import asyncio

from src.trainer import Trainer, TrainerConfig


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
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Max new tokens per generation (default: 200)")

    # Model settings
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model to train")
    parser.add_argument("--judge_model", type=str, default="qwen2.5:7b",
                        help="Ollama model for judging")

    # Paths
    parser.add_argument("--chroma_db_dir", type=str, default="data/.chroma_db",
                        help="ChromaDB directory")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints",
                        help="Output directory for checkpoints")

    return parser.parse_args()


async def main_async() -> None:
    """Async main function."""
    args = parse_args()

    config = TrainerConfig(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        max_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model_name,
        judge_model=args.judge_model,
        chroma_db_dir=args.chroma_db_dir,
        output_dir=args.output_dir,
    )

    trainer = Trainer(config)
    trainer.setup()
    await trainer.train()


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
