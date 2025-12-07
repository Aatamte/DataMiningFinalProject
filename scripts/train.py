"""CLI entrypoint for training.

Requires TRAIN_CONFIG environment variable pointing to a YAML config file.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --eval-only
    uv run python scripts/train.py --live-plot
"""

import argparse
import asyncio
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import load_config, get_config_value
from src.trainer import Trainer, TrainerConfig


async def check_sandbox() -> tuple[bool, str]:
    """Test sandbox execution with a simple code snippet."""
    from src.environment import SandboxClient
    try:
        async with SandboxClient() as sandbox:
            result = await sandbox.execute("print(2 + 2)")
            if result.get("error"):
                return False, f"Execution error: {result['error']}"
            if "4" in result.get("output", ""):
                return True, "Sandbox executed test code successfully"
            return False, f"Unexpected output: {result.get('output', '')}"
    except Exception as e:
        return False, str(e)


async def validate_environment(config: dict) -> bool:
    """Validate that required services are reachable.

    Returns True if all checks pass, False otherwise.
    """
    judge_url = get_config_value(config, "judge", "base_url")
    judge_model = get_config_value(config, "judge", "model")
    model_name = get_config_value(config, "model", "name")

    print("=" * 60)
    print("VALIDATING ENVIRONMENT")
    print("=" * 60)

    all_ok = True

    # Check judge API is reachable
    print(f"\n[1/3] Checking judge API: {judge_url}")
    try:
        resp = httpx.get(f"{judge_url}/models", timeout=30.0)
        if resp.status_code == 200:
            models = resp.json()
            model_ids = [m.get("id", "") for m in models.get("data", [])]
            print(f"  [OK] API reachable, available models: {model_ids}")

            # Check if judge model is available
            if judge_model in model_ids or any(judge_model in m for m in model_ids):
                print(f"  [OK] Judge model '{judge_model}' available")
            else:
                print(f"  [FAIL] Judge model '{judge_model}' not found in available models")
                print(f"    Available: {model_ids}")
                all_ok = False
        else:
            print(f"  [FAIL] API returned status {resp.status_code}")
            all_ok = False
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to {judge_url}")
        print("    Is LM Studio / Ollama running?")
        all_ok = False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        all_ok = False

    # Check training model exists on HuggingFace (just validate format)
    print(f"\n[2/3] Checking training model: {model_name}")
    if "/" in model_name:
        print("  [OK] Model name format valid (will download if needed)")
    else:
        print(f"  ! Warning: Model name '{model_name}' may be local path")

    # Check sandbox execution
    print(f"\n[3/3] Checking sandbox execution...")
    try:
        sandbox_ok, sandbox_msg = await check_sandbox()
        if sandbox_ok:
            print(f"  [OK] {sandbox_msg}")
        else:
            print(f"  [FAIL] {sandbox_msg}")
            print("    Is the Docker sandbox running? Try: uv run python scripts/run_environment.py")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        all_ok = False

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

    # Runtime flags only - config comes from YAML
    parser.add_argument("--eval-only", action="store_true",
                        help="Run episodes without training (no weight updates)")
    parser.add_argument("--live-plot", action="store_true",
                        help="Update plot after each step")

    return parser.parse_args()


async def main_async() -> None:
    """Async main function."""
    args = parse_args()

    # Load config from YAML (fails if TRAIN_CONFIG not set)
    config = load_config()

    # Validate environment before starting
    if not await validate_environment(config):
        sys.exit(1)

    # Build TrainerConfig from YAML
    trainer_config = TrainerConfig.from_yaml(config)
    trainer_config.eval_only = args.eval_only

    trainer = Trainer(trainer_config)
    trainer.setup()
    await trainer.train(live_plot=args.live_plot)


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Cleaning up...")
        # Force cleanup of any lingering GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    main()
