"""Judge prompt for evaluating answers."""

import json

# JSON Schema for judge response - modify this to change what the judge returns
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "Whether the response correctly answers the question"
        }
    },
    "required": ["correct"]
}

JUDGE_PROMPT_TEMPLATE = """Does the response contain the ground truth answer?

Question: {question}
Ground truth: {answer}
Response: {response}

STRICT RULES:
- "correct": true ONLY if the response contains "{answer}" or equivalent
- "correct": false if response is code, "not found", "no information", or missing the answer

{schema}

JSON only:"""


def build_judge_prompt(question: str, answer: str, response: str, schema: dict | None = None) -> str:
    """Build the judge prompt with filled values.

    Args:
        question: The question being answered
        answer: Ground truth answer
        response: Model's response to evaluate
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
