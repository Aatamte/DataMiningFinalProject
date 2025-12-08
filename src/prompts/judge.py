"""Judge prompt for evaluating answers and approach quality."""

import json
import re

# JSON Schema for judge response - uses categories for clearer classification
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["correct", "incorrect", "gave_up"],
            "description": "Classification of the response"
        },
        "approach_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Quality of the search approach (0-100)"
        }
    },
    "required": ["category", "approach_score"]
}

# Simple schema - category only (faster, fewer tokens)
JUDGE_SCHEMA_SIMPLE = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["correct", "incorrect", "gave_up"],
            "description": "Classification of the response"
        }
    },
    "required": ["category"]
}

# Schema for batch judge response (array of results)
BATCH_JUDGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "string", "enum": ["correct", "incorrect", "gave_up"]},
            "approach_score": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["id", "category", "approach_score"]
    }
}

# Simple batch schema - category only
BATCH_JUDGE_SCHEMA_SIMPLE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "string", "enum": ["correct", "incorrect", "gave_up"]}
        },
        "required": ["id", "category"]
    }
}

JUDGE_PROMPT_TEMPLATE = """Evaluate the agent's answer AND search approach.

Question: {question}
Ground truth answer: {answer}
Agent's final answer: {response}

Agent's full trajectory:
{trajectory}

SCORING:

1. "category" (one of: "correct", "incorrect", "gave_up"):
   - "correct": Answer is semantically equivalent to "{answer}" (same meaning, different wording OK)
     Examples: "42 years old" matches "42", "NYC" matches "New York City"
   - "incorrect": Agent provided a WRONG answer (tried but got it wrong)
   - "gave_up": Agent gave up without providing a real answer
     Examples: "Unknown", "Not found", "N/A", "Unable to determine", "I don't know"

   IMPORTANT: "gave_up" is WORSE than "incorrect" - always penalize giving up!

2. "approach_score" (0-100):
   - 0-25: Relevant search queries (targeted vs generic)
   - 0-25: Efficiency (minimal wasted turns, no repeated failures)
   - 0-25: Found and read appropriate sources
   - 0-25: Clear reasoning, answer derived from evidence

Think briefly, then output JSON.

{schema}

JSON:"""

# Simple prompt - category only (faster, fewer tokens)
JUDGE_PROMPT_TEMPLATE_SIMPLE = """Classify the agent's answer.

Question: {question}
Ground truth: {answer}
Agent's answer: {response}

Categories:
- "correct": Answer matches ground truth (semantically equivalent)
- "incorrect": Wrong answer (tried but failed)
- "gave_up": Gave up ("Unknown", "Not found", "N/A", etc.) - WORST category!

Output: {{"category": "correct"}}, {{"category": "incorrect"}}, or {{"category": "gave_up"}}

JSON:"""


def build_judge_prompt(
    question: str,
    answer: str,
    response: str,
    trajectory: str = "",
    schema: dict | None = None,
    simple: bool = False,
) -> str:
    """Build the judge prompt with filled values.

    Args:
        question: The question being answered
        answer: Ground truth answer
        response: Model's final answer to evaluate
        trajectory: Full conversation trajectory (code, outputs, reasoning)
        schema: JSON schema for response format (uses JUDGE_SCHEMA if None)
        simple: If True, use simple correctness-only prompt (faster, fewer tokens)

    Returns:
        Formatted prompt string
    """
    if simple:
        return JUDGE_PROMPT_TEMPLATE_SIMPLE.format(
            question=question,
            answer=answer,
            response=response,
        )

    if schema is None:
        schema = JUDGE_SCHEMA

    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        response=response,
        trajectory=trajectory or "(no trajectory provided)",
        schema=json.dumps(schema, indent=2)
    )


def parse_judge_response(response: str, schema: dict | None = None, simple: bool = False) -> dict:
    """Parse and validate judge response against schema.

    Args:
        response: Raw response string from judge
        schema: Expected schema (uses JUDGE_SCHEMA if None)
        simple: If True, only require 'category' field, default approach_score to 0

    Returns:
        Parsed JSON dict with 'category', 'correct', and 'approach_score' keys

    Raises:
        ValueError: If response is not valid JSON or missing required fields
    """
    import re

    if schema is None:
        schema = JUDGE_SCHEMA_SIMPLE if simple else JUDGE_SCHEMA

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

    # Handle category field - convert to correct boolean for backwards compatibility
    if "category" in data:
        category = data["category"].lower()
        data["correct"] = (category == "correct")
        data["gave_up"] = (category == "gave_up")
    elif "correct" in data:
        # Backwards compatibility: if old format with "correct" boolean
        data["category"] = "correct" if data["correct"] else "incorrect"
        data["gave_up"] = False

    # Default approach_score to 0 if not present (simple mode)
    if "approach_score" not in data:
        data["approach_score"] = 0

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

1. "category" (one of: "correct", "incorrect", "gave_up"):
   - "correct": Answer is semantically equivalent to "{answer}"
   - "incorrect": Agent provided a WRONG answer (tried but got it wrong)
   - "gave_up": Agent gave up ("Unknown", "Not found", "N/A", etc.) - WORST!

2. "approach_score" (0-100):
   - 0-25: Relevant search queries (targeted vs generic)
   - 0-25: Efficiency (minimal wasted turns, no repeated failures)
   - 0-25: Found and read appropriate sources
   - 0-25: Clear reasoning, answer derived from evidence

IMPORTANT:
- "gave_up" is ALWAYS worse than "incorrect" - penalize it!
- NEVER give two agents the same approach_score. Rank them.
- Output a JSON array with one object per ID.

Example output format:
[
  {{"id": 0, "category": "correct", "approach_score": 85}},
  {{"id": 1, "category": "incorrect", "approach_score": 40}},
  {{"id": 2, "category": "gave_up", "approach_score": 10}}
]

JSON:"""

# Simple batch prompt - category only
BATCH_JUDGE_PROMPT_TEMPLATE_SIMPLE = """Classify these answers.

Question: {question}
Ground truth: {answer}

{entries}

Categories:
- "correct": Answer matches ground truth
- "incorrect": Wrong answer (tried but failed)
- "gave_up": Gave up ("Unknown", "Not found", etc.) - WORST!

Output JSON array:
[{{"id": 0, "category": "correct"}}, {{"id": 1, "category": "incorrect"}}, {{"id": 2, "category": "gave_up"}}]

JSON:"""


def build_batch_judge_prompt(
    question: str,
    answer: str,
    entries: list[dict],
    simple: bool = False,
) -> str:
    """Build batch judge prompt for multiple answers.

    Args:
        question: The question being answered
        answer: Ground truth answer
        entries: List of dicts with keys: id, response, trajectory
        simple: If True, use simple correctness-only prompt (faster, fewer tokens)

    Returns:
        Formatted prompt string
    """
    entry_texts = []
    for entry in entries:
        entry_id = entry["id"]
        response = entry.get("response", "(no answer)")

        if simple:
            # Simple mode: just ID and answer
            entry_text = f"[ID: {entry_id}] Answer: {response}"
        else:
            # Full mode: include trajectory
            trajectory = entry.get("trajectory", "(no trajectory)")
            entry_text = f"""---
[ID: {entry_id}]
Final answer: {response}
Trajectory:
{trajectory}
"""
        entry_texts.append(entry_text)

    template = BATCH_JUDGE_PROMPT_TEMPLATE_SIMPLE if simple else BATCH_JUDGE_PROMPT_TEMPLATE
    return template.format(
        question=question,
        answer=answer,
        entries="\n".join(entry_texts),
    )


def parse_batch_judge_response(response: str, simple: bool = False) -> list[dict]:
    """Parse batch judge response into list of results.

    Args:
        response: Raw response string from judge
        simple: If True, approach_score is optional (defaults to 0)

    Returns:
        List of dicts with keys: id, category, correct, gave_up, approach_score

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

        # Handle category or correct field
        if "category" in item:
            category = item["category"].lower()
            correct = (category == "correct")
            gave_up = (category == "gave_up")
        elif "correct" in item:
            # Backwards compatibility
            correct = bool(item["correct"])
            category = "correct" if correct else "incorrect"
            gave_up = False
        else:
            raise ValueError(f"Missing 'category' or 'correct' field in: {item}")

        if not simple and "approach_score" not in item:
            raise ValueError(f"Missing 'approach_score' field in: {item}")

        results.append({
            "id": item["id"],
            "category": category,
            "correct": correct,
            "gave_up": gave_up,
            "approach_score": int(item.get("approach_score", 0)),
        })

    return results
