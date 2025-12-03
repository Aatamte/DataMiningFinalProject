"""Parsing utilities for tool calls and answers."""

import re


def parse_tool_call(text: str) -> tuple[str | None, str | None]:
    """Parse a tool call from model output.

    Expected formats:
        - <tool>tool_name(arg1, arg2)</tool>
        - tool_name("arg")
        - tool_name: query

    Args:
        text: Model output text to parse

    Returns:
        Tuple of (function_name, arguments) or (None, None) if no valid call found
    """
    # Try XML-style format
    match = re.search(r'<tool>(\w+)\(([^)]*)\)</tool>', text)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        return func_name, args_str

    # Try function-style format
    match = re.search(r'(\w+)\("([^"]*)"\)', text)
    if match:
        return match.group(1), match.group(2)

    # Try simple format: search_pages: query
    match = re.search(
        r'(search_pages|view_sections|read_section)[:\s]+["\']?([^"\'<\n]+)["\']?',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1), match.group(2).strip()

    return None, None


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
