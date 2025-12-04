"""Trainer subpackage for SLM-RL Search.

Lightweight imports (no torch dependency):
    from src.trainer.parsing import parse_python_code, parse_answer
    from src.trainer.metrics import MetricsTracker, setup_logging

Full imports (requires torch):
    from src.trainer import Trainer, TrainerConfig
"""

# Lightweight imports - always available
from .parsing import parse_python_code, parse_answer
from .metrics import MetricsTracker, setup_logging


def __getattr__(name):
    """Lazy import heavy dependencies."""
    if name == "Trainer":
        from .core import Trainer
        return Trainer
    if name == "TrainerConfig":
        from .core import TrainerConfig
        return TrainerConfig
    if name == "get_judge_reward":
        from .episode import get_judge_reward
        return get_judge_reward
    if name == "JudgeResult":
        from .episode import JudgeResult
        return JudgeResult
    if name == "compute_reinforce_loss":
        from .episode import compute_reinforce_loss
        return compute_reinforce_loss
    if name == "EpisodeLogger":
        from .episode_logger import EpisodeLogger
        return EpisodeLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Trainer",
    "TrainerConfig",
    "MetricsTracker",
    "setup_logging",
    "get_judge_reward",
    "compute_reinforce_loss",
    "parse_python_code",
    "parse_answer",
    "EpisodeLogger",
]
