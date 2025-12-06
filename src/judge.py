"""Local judge model wrapper using OpenAI-compatible API."""

import os

from openai import AsyncOpenAI, OpenAI

from .prompts import JUDGE_PROMPT

# Re-export for backwards compatibility
__all__ = ["get_local_judge_client", "get_sync_judge_client", "JUDGE_PROMPT"]

# Default to LM Studio port, override with JUDGE_BASE_URL env var
DEFAULT_JUDGE_URL = "http://localhost:1234/v1"


def get_local_judge_client(
    base_url: str | None = None,
) -> AsyncOpenAI:
    """Get an AsyncOpenAI client pointing to local LLM server.

    Works with any OpenAI-compatible API (LM Studio, Ollama, etc.)

    Args:
        base_url: URL of the API. Defaults to JUDGE_BASE_URL env var,
                  or http://localhost:1234/v1 (LM Studio)

    Returns:
        AsyncOpenAI client configured for local inference.
    """
    if base_url is None:
        base_url = os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_URL)

    return AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",
    )


def get_sync_judge_client(
    base_url: str | None = None,
) -> OpenAI:
    """Get a sync OpenAI client for thread-based judge calls.

    Args:
        base_url: URL of the API. Defaults to JUDGE_BASE_URL env var,
                  or http://localhost:1234/v1 (LM Studio)

    Returns:
        OpenAI client configured for local inference (sync).
    """
    if base_url is None:
        base_url = os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_URL)

    return OpenAI(
        base_url=base_url,
        api_key="not-needed",
    )
