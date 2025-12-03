"""Trainer subpackage for SLM-RL Search.

Lightweight imports (no torch dependency):
    from src.trainer.parsing import parse_tool_call, parse_answer
    from src.trainer.metrics import MetricsTracker, setup_logging

Full imports (requires torch):
    from src.trainer import Trainer, TrainerConfig
"""

# Lightweight imports - always available
from .parsing import parse_tool_call, parse_answer
from .metrics import MetricsTracker, setup_logging


def __getattr__(name):
    """Lazy import heavy dependencies."""
    if name == "Trainer":
        from .core import Trainer
        return Trainer
    if name == "TrainerConfig":
        from .core import TrainerConfig
        return TrainerConfig
    if name == "run_episode":
        from .episode import run_episode
        return run_episode
    if name == "get_judge_reward":
        from .episode import get_judge_reward
        return get_judge_reward
    if name == "compute_reinforce_loss":
        from .episode import compute_reinforce_loss
        return compute_reinforce_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Trainer",
    "TrainerConfig",
    "MetricsTracker",
    "setup_logging",
    "run_episode",
    "get_judge_reward",
    "compute_reinforce_loss",
    "parse_tool_call",
    "parse_answer",
]
