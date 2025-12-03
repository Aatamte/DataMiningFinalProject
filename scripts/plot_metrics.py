#!/usr/bin/env python3
"""Generate plots from training metrics.

Usage:
    uv run python scripts/plot_metrics.py runs/train_20241203_143022
    uv run python scripts/plot_metrics.py runs/train_20241203_143022 --output plots.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> dict:
    """Load metrics from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file) as f:
        return json.load(f)


def plot_metrics(metrics: dict, output_path: Path | None = None, show: bool = True):
    """Generate plots from metrics.

    Args:
        metrics: Metrics dict from training
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Metrics", fontsize=14)

    # Plot 1: Loss over steps
    ax1 = axes[0, 0]
    if metrics.get("losses"):
        ax1.plot(metrics["steps"], metrics["losses"], "b-", linewidth=1)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss per Step")
        ax1.grid(True, alpha=0.3)

    # Plot 2: Reward over steps
    ax2 = axes[0, 1]
    if metrics.get("rewards"):
        ax2.plot(metrics["steps"], metrics["rewards"], "g-", linewidth=1)
        ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Baseline")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Avg Reward")
        ax2.set_title("Reward per Step")
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Epoch average reward
    ax3 = axes[1, 0]
    if metrics.get("epoch_avg_rewards"):
        epochs = list(range(1, len(metrics["epoch_avg_rewards"]) + 1))
        ax3.bar(epochs, metrics["epoch_avg_rewards"], color="purple", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Avg Reward")
        ax3.set_title("Average Reward per Epoch")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Rolling average reward
    ax4 = axes[1, 1]
    if metrics.get("rewards") and len(metrics["rewards"]) > 1:
        rewards = metrics["rewards"]
        window = min(10, len(rewards))
        rolling_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling_avg.append(sum(rewards[start:i+1]) / (i - start + 1))

        ax4.plot(metrics["steps"], rewards, "g-", alpha=0.3, label="Raw")
        ax4.plot(metrics["steps"], rolling_avg, "g-", linewidth=2, label=f"Rolling avg (w={window})")
        ax4.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Reward")
        ax4.set_title("Reward with Rolling Average")
        ax4.set_ylim(-0.1, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        return 1

    metrics = load_metrics(args.run_dir)

    output_path = args.output
    if output_path is None:
        output_path = args.run_dir / "plots.png"

    plot_metrics(metrics, output_path=output_path, show=not args.no_show)
    return 0


if __name__ == "__main__":
    exit(main())
