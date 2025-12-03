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

JUDGE_PROMPT_TEMPLATE = """Given a ground truth answer and a response, determine if the response is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond with JSON matching this schema:
```json
{schema}
```

If a response contains incoherent text, set "correct" to false even if the correct answer is also present.

Respond with valid JSON only, no other text."""


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
    if schema is None:
        schema = JUDGE_SCHEMA

    # Try to extract JSON from response
    response = response.strip()

    # Handle markdown code blocks
    if response.startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return data


# Legacy export for backwards compatibility
JUDGE_PROMPT = JUDGE_PROMPT_TEMPLATE
