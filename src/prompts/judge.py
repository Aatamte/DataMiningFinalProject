"""Judge prompt for evaluating answers and approach quality."""

import json
import re

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

# Schema for batch judge response (array of results)
BATCH_JUDGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "correct": {"type": "boolean"},
            "approach_score": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["id", "correct", "approach_score"]
    }
}

JUDGE_PROMPT_TEMPLATE = """Evaluate the agent's answer AND search approach.

Question: {question}
Ground truth answer: {answer}
Agent's final answer: {response}

Agent's full trajectory:
{trajectory}

SCORING:

1. "correct" (true/false):
   - true ONLY if final answer is EXACTLY "{answer}" or nearly identical (minor formatting OK)
   - false if partial, abbreviated, missing key words, "not found", or wrong
   - Be STRICT: "Chabert Collection" is NOT correct if ground truth is "Lacey Chabert Collection"

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

    # Strip <think>...</think> tags first (deepseek-r1 models use these)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Just find the JSON object, ignore everything else (markdown, etc.)
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


# =============================================================================
# Batch Judge Prompt (multiple answers in one call)
# =============================================================================

BATCH_JUDGE_PROMPT_TEMPLATE = """Evaluate multiple agent responses to the same question.

Question: {question}
Ground truth answer: {answer}

{entries}

SCORING CRITERIA:

1. "correct" (true/false):
   - true ONLY if final answer is EXACTLY "{answer}" or nearly identical (minor formatting OK)
   - false if partial, abbreviated, missing key words, "not found", or wrong
   - Be STRICT: "Chabert Collection" is NOT correct if ground truth is "Lacey Chabert Collection"

2. "approach_score" (0-100):
   - 0-25: Relevant search queries (targeted vs generic)
   - 0-25: Efficiency (minimal wasted turns, no repeated failures)
   - 0-25: Found and read appropriate sources
   - 0-25: Clear reasoning, answer derived from evidence

IMPORTANT:
- NEVER give two agents the same approach_score. Rank them - one must be better.
- Output a JSON array with one object per ID. Think briefly, then output JSON.

Example output format:
[
  {{"id": 0, "correct": true, "approach_score": 85}},
  {{"id": 1, "correct": false, "approach_score": 40}}
]

JSON:"""


def build_batch_judge_prompt(
    question: str,
    answer: str,
    entries: list[dict],
) -> str:
    """Build batch judge prompt for multiple answers.

    Args:
        question: The question being answered
        answer: Ground truth answer
        entries: List of dicts with keys: id, response, trajectory

    Returns:
        Formatted prompt string
    """
    entry_texts = []
    for entry in entries:
        entry_id = entry["id"]
        response = entry.get("response", "(no answer)")
        trajectory = entry.get("trajectory", "(no trajectory)")

        entry_text = f"""---
[ID: {entry_id}]
Final answer: {response}
Trajectory:
{trajectory}
"""
        entry_texts.append(entry_text)

    return BATCH_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        entries="\n".join(entry_texts),
    )


def parse_batch_judge_response(response: str) -> list[dict]:
    """Parse batch judge response into list of results.

    Args:
        response: Raw response string from judge

    Returns:
        List of dicts with keys: id, correct, approach_score

    Raises:
        ValueError: If response is not valid JSON array or missing required fields
    """
    original_response = response

    # Strip <think>...</think> tags (deepseek-r1 models)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Find JSON array - look for [...] pattern
    match = re.search(r"\[[\s\S]*?\]", response)
    if match:
        response = match.group(0)
    else:
        response = response.strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON array: {e}\nOriginal response:\n{original_response[:1500]}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    # Validate each entry
    results = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Expected object in array, got {type(item).__name__}")

        if "id" not in item:
            raise ValueError(f"Missing 'id' field in: {item}")
        if "correct" not in item:
            raise ValueError(f"Missing 'correct' field in: {item}")
        if "approach_score" not in item:
            raise ValueError(f"Missing 'approach_score' field in: {item}")

        results.append({
            "id": item["id"],
            "correct": bool(item["correct"]),
            "approach_score": int(item["approach_score"]),
        })

    return results
