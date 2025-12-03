"""Environment subpackage for wiki-search.

Lightweight imports (no heavy dependencies):
    from src.environment.sandbox import SandboxClient, SandboxError
    from src.environment.tools import normalize_id

Full imports (requires chromadb, datasets, etc.):
    from src.environment import load_environment
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
from .tools import normalize_id


def __getattr__(name):
    """Lazy import heavy dependencies."""
    if name == "load_environment":
        from .core import load_environment
        return load_environment
    if name == "create_search_tools":
        from .tools import create_search_tools
        return create_search_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_environment",
    "create_search_tools",
    "normalize_id",
    "SandboxClient",
    "SandboxError",
    "start_server",
    "stop_server",
    "is_server_running",
    "execute_code",
]
