"""Prompt templates for SLM-RL Search."""

from .system import SYSTEM_PROMPT
from .judge import (
    JUDGE_PROMPT,
    JUDGE_SCHEMA,
    build_judge_prompt,
    parse_judge_response,
)

__all__ = [
    "SYSTEM_PROMPT",
    "JUDGE_PROMPT",
    "JUDGE_SCHEMA",
    "build_judge_prompt",
    "parse_judge_response",
]
