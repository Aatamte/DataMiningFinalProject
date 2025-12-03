"""Metrics tracking and logging utilities for training."""

import json
import logging
from pathlib import Path


def setup_run_dir(run_name: str, base_dir: str = "runs") -> Path:
    """Create run directory structure.

    Args:
        run_name: Name for this training run
        base_dir: Base directory for all runs

    Returns:
        Path to the run directory
    """
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> tuple[logging.Logger, Path]:
    """Set up logging to console and file.

    Args:
        run_dir: Directory for this run

    Returns:
        Tuple of (logger, log_file_path)
    """
    log_file = run_dir / "train.log"

    # Create logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler (more detailed)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


class MetricsTracker:
    """Track training metrics across steps and epochs."""

    def __init__(self, run_dir: Path):
        """Initialize metrics tracker.

        Args:
            run_dir: Directory for this run
        """
        self.run_dir = run_dir
        self.metrics = {
            "steps": [],
            "losses": [],
            "rewards": [],
            "epoch_avg_rewards": [],
        }
        self.step = 0

    def log_step(self, loss: float, rewards: list):
        """Log metrics for a training step.

        Args:
            loss: Loss value for this step
            rewards: List of rewards from rollouts
        """
        self.step += 1
        self.metrics["steps"].append(self.step)
        self.metrics["losses"].append(loss)
        self.metrics["rewards"].append(sum(rewards) / len(rewards) if rewards else 0)

    def log_epoch(self, avg_reward: float):
        """Log epoch-level metrics.

        Args:
            avg_reward: Average reward for the epoch
        """
        self.metrics["epoch_avg_rewards"].append(avg_reward)

    def save(self) -> Path:
        """Save metrics to JSON file.

        Returns:
            Path to the saved metrics file
        """
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return metrics_file
