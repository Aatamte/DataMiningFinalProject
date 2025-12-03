"""REINFORCE training utilities."""

import time

import torch

from src.prompts import build_judge_prompt, parse_judge_response
from src.llm_logger import get_llm_logger
from src.agent import EpisodeResult


async def get_judge_reward(
    judge_client,
    judge_model: str,
    question: str,
    answer: str,
    response: str
) -> float:
    """Get reward from judge model.

    Args:
        judge_client: OpenAI-compatible async client for judge
        judge_model: Model name to use for judging
        question: The original question
        answer: Ground truth answer
        response: Model's response to evaluate

    Returns:
        Reward value (1.0 for correct, 0.0 for incorrect)
    """
    prompt = build_judge_prompt(question, answer, response)
    messages = [{"role": "user", "content": prompt}]

    try:
        start_time = time.perf_counter()
        completion = await judge_client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=50,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        judge_response = completion.choices[0].message.content
        result = parse_judge_response(judge_response)
        reward = 1.0 if result.get("correct") else 0.0

        # Log judge call
        llm_logger = get_llm_logger()
        if llm_logger:
            usage = None
            if hasattr(completion, "usage") and completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                }
            llm_logger.log_judge_call(
                model=judge_model,
                messages=messages,
                response_content=judge_response,
                max_tokens=50,
                temperature=0,
                latency_ms=latency_ms,
                parsed_result=result,
                usage=usage,
                metadata={"question": question, "answer": answer, "model_response": response},
            )

        return reward
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0.0


def compute_reinforce_loss(
    model,
    tokenizer,
    episodes: list[EpisodeResult],
    rewards: list[float],
    device: str,
    baseline: float = 0.5
) -> torch.Tensor:
    """Compute REINFORCE loss for policy gradient update.

    Uses fixed baseline of 0.5 (midpoint for binary rewards) to ensure
    gradients even when all rewards are the same.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        episodes: List of EpisodeResult from agent runs
        rewards: List of rewards corresponding to each episode
        device: Device to compute on
        baseline: Fixed baseline for advantage computation

    Returns:
        Computed loss tensor
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_steps = 0

    for episode, reward in zip(episodes, rewards):
        advantage = reward - baseline

        # Get trajectory in (prompt, response) format
        trajectory = episode.trajectory()

        # For each turn in the trajectory, compute loss
        for prompt, response in trajectory:
            full_text = prompt + response
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)

            # Get log probabilities
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Weight by advantage (REINFORCE)
            # positive advantage = reward > baseline = reinforce this behavior
            # negative advantage = reward < baseline = discourage this behavior
            weighted_loss = loss * (-advantage)
            total_loss = total_loss + weighted_loss
            n_steps += 1

    return total_loss / max(n_steps, 1)
