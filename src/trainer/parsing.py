"""Parsing utilities for Python code and answers."""

import re


def parse_python_code(text: str) -> str | None:
    """Parse Python code from model output.

    Expected format: <python>...code...</python>

    Args:
        text: Model output text to parse

    Returns:
        The Python code string if found, None otherwise
    """
    match = re.search(r'<python>(.*?)</python>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_answer(text: str) -> str | None:
    """Parse an answer from model output.

    Expected format: <answer>...</answer>

    Args:
        text: Model output text to parse

    Returns:
        The answer string if found, None otherwise
    """
    match = re.search(r'<answer>([^<]+)</answer>', text)
    if match:
        return match.group(1).strip()
    return None
