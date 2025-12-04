"""Judge prompt for evaluating answers and approach quality."""

import json

# JSON Schema for judge response - modify this to change what the judge returns
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "Whether the response correctly answers the question"
        },
        "approach_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Quality of the search approach (0-100)"
        }
    },
    "required": ["correct", "approach_score"]
}

JUDGE_PROMPT_TEMPLATE = """Evaluate the agent's answer AND search approach.

Question: {question}
Ground truth answer: {answer}
Agent's final answer: {response}

Agent's full trajectory:
{trajectory}

SCORING:

1. "correct" (true/false):
   - true ONLY if final answer contains "{answer}" or semantic equivalent
   - false if answer is wrong, missing, "not found", or just code

2. "approach_score" (0-100):
   - 0-25: Relevant search queries (targeted vs generic)
   - 0-25: Efficiency (minimal wasted turns, no repeated failures)
   - 0-25: Found and read appropriate sources
   - 0-25: Clear reasoning, answer derived from evidence

   Examples:
   - 90-100: Targeted search, found info quickly, clear reasoning
   - 60-80: Good search but some wasted turns
   - 30-50: Eventually found answer but inefficient
   - 0-20: Random searching, no clear strategy, or hallucinated

IMPORTANT: Think briefly (under 100 words), then output JSON immediately.

{schema}

JSON:"""


def build_judge_prompt(
    question: str,
    answer: str,
    response: str,
    trajectory: str = "",
    schema: dict | None = None
) -> str:
    """Build the judge prompt with filled values.

    Args:
        question: The question being answered
        answer: Ground truth answer
        response: Model's final answer to evaluate
        trajectory: Full conversation trajectory (code, outputs, reasoning)
        schema: JSON schema for response format (uses JUDGE_SCHEMA if None)

    Returns:
        Formatted prompt string
    """
    if schema is None:
        schema = JUDGE_SCHEMA

    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        response=response,
        trajectory=trajectory or "(no trajectory provided)",
        schema=json.dumps(schema, indent=2)
    )


def parse_judge_response(response: str, schema: dict | None = None) -> dict:
    """Parse and validate judge response against schema.

    Args:
        response: Raw response string from judge
        schema: Expected schema (uses JUDGE_SCHEMA if None)

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If response is not valid JSON or missing required fields
    """
    import re

    if schema is None:
        schema = JUDGE_SCHEMA

    original_response = response  # Keep for error reporting

    # Just find the JSON object, ignore everything else (thinking tags, markdown, etc.)
    match = re.search(r"\{[^{}]*\}", response)
    if match:
        response = match.group(0)
    else:
        response = response.strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nOriginal response:\n{original_response[:1000]}")

    # Validate required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return data


# Legacy export for backwards compatibility
JUDGE_PROMPT = JUDGE_PROMPT_TEMPLATE
