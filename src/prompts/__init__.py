"""Prompt templates for SLM-RL Search."""

from .system import get_system_prompt
from .judge import (
    JUDGE_PROMPT,
    JUDGE_SCHEMA,
    build_judge_prompt,
    parse_judge_response,
    # Batch judge
    BATCH_JUDGE_SCHEMA,
    build_batch_judge_prompt,
    parse_batch_judge_response,
)

__all__ = [
    "get_system_prompt",
    "JUDGE_PROMPT",
    "JUDGE_SCHEMA",
    "build_judge_prompt",
    "parse_judge_response",
    # Batch judge
    "BATCH_JUDGE_SCHEMA",
    "build_batch_judge_prompt",
    "parse_batch_judge_response",
]
