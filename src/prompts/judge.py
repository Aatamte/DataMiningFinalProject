"""Judge prompt for evaluating answers."""

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is both correct and coherent.

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

Respond either "yes" or "no" only.

If a response contains incoherent text, respond with "no" even if the correct answer is also present."""


def build_judge_prompt(question: str, answer: str, response: str) -> str:
    """Build the judge prompt with filled values."""
    return JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        response=response
    )
