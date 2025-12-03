"""Environment subpackage for wiki-search.

Lightweight imports (no heavy dependencies):
    from src.environment import SandboxClient, SandboxError

Full imports (requires datasets):
    from src.environment import load_questions
"""

# Lightweight imports - always available
from .sandbox import (
    SandboxClient,
    SandboxError,
    start_server,
    stop_server,
    is_server_running,
    execute_code,
)


def __getattr__(name):
    """Lazy import heavy dependencies."""
    if name == "load_questions":
        from .core import load_questions
        return load_questions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_questions",
    "SandboxClient",
    "SandboxError",
    "start_server",
    "stop_server",
    "is_server_running",
    "execute_code",
]
