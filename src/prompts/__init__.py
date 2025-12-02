"""Prompt templates for SLM-RL Search."""

from .system import SYSTEM_PROMPT, build_prompt
from .judge import JUDGE_PROMPT

__all__ = ["SYSTEM_PROMPT", "JUDGE_PROMPT", "build_prompt"]
