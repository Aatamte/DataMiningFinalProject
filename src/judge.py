"""Local judge model wrapper using Ollama."""

from openai import AsyncOpenAI

from .prompts import JUDGE_PROMPT

# Re-export for backwards compatibility
__all__ = ["get_local_judge_client", "JUDGE_PROMPT"]


def get_local_judge_client(
    base_url: str = "http://localhost:11434/v1",
) -> AsyncOpenAI:
    """Get an AsyncOpenAI client pointing to local Ollama instance.

    Ollama exposes an OpenAI-compatible API, so we can use the standard
    AsyncOpenAI client with a different base URL.

    Args:
        base_url: URL of the Ollama API (default: localhost:11434/v1)

    Returns:
        AsyncOpenAI client configured for Ollama.
    """
    return AsyncOpenAI(
        base_url=base_url,
        api_key="ollama",  # Dummy key, not used by Ollama
    )
