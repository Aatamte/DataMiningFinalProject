"""Metrics tracking and logging utilities for training."""

import json
import logging
import os


def setup_logging(output_dir: str, run_name: str) -> tuple[logging.Logger, str]:
    """Set up logging to console and file.

    Args:
        output_dir: Directory for log files
        run_name: Name for this training run

    Returns:
        Tuple of (logger, log_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"{run_name}.log")

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

    def __init__(self, output_dir: str, run_name: str):
        """Initialize metrics tracker.

        Args:
            output_dir: Directory for saving metrics
            run_name: Name for this training run
        """
        self.output_dir = output_dir
        self.run_name = run_name
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

    def save(self) -> str:
        """Save metrics to JSON file.

        Returns:
            Path to the saved metrics file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_file = os.path.join(self.output_dir, f"{self.run_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return metrics_file
