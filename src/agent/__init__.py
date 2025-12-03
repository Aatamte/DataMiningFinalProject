"""Agent module for conversation management and episode execution.

Lightweight imports (no torch dependency):
    from src.agent import Message, Role, Conversation, EpisodeResult

Full imports (requires torch):
    from src.agent import Agent, AgentConfig
"""

# Lightweight imports - always available
from .message import Message, Role
from .conversation import Conversation
from .episode import EpisodeResult


def __getattr__(name):
    """Lazy import heavy dependencies."""
    if name == "Agent":
        from .agent import Agent
        return Agent
    if name == "AgentConfig":
        from .agent import AgentConfig
        return AgentConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Message",
    "Role",
    "Conversation",
    "Agent",
    "AgentConfig",
    "EpisodeResult",
]
