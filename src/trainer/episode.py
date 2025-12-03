"""Episode execution and REINFORCE training utilities."""

import json

import torch

from src.prompts import JUDGE_PROMPT, build_prompt
from .parsing import parse_tool_call, parse_answer


async def run_episode(
    model,
    tokenizer,
    question: str,
    answer: str,
    tools: list,
    tool_dict: dict,
    max_turns: int,
    max_new_tokens: int,
    device: str,
    max_context: int = 1500
) -> tuple[list, str]:
    """Run a single episode: model uses tools to answer a question.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        question: The question to answer
        answer: Ground truth answer (unused during episode, for logging)
        tools: List of tool functions
        tool_dict: Dict mapping tool names to functions
        max_turns: Maximum tool-use turns
        max_new_tokens: Max tokens to generate per turn
        device: Device to run on (cuda/cpu)
        max_context: Maximum context length in tokens

    Returns:
        Tuple of (trajectory, final_answer) where trajectory is list of (prompt, response) pairs
    """
    conversation = build_prompt(question)
    initial_len = len(conversation)
    trajectory = []  # Store (prompt, response) pairs

    for turn in range(max_turns):
        # Truncate conversation if too long (keep system prompt + recent context)
        if len(conversation) > max_context * 4:  # rough char estimate
            # Keep initial prompt and last portion
            conversation = conversation[:initial_len] + "\n...(truncated)...\n" + conversation[-(max_context * 2):]

        # Generate response
        inputs = tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=max_context
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(conversation):]

        trajectory.append((conversation, response))

        # Check for final answer
        final_answer = parse_answer(response)
        if final_answer is not None:
            return trajectory, final_answer

        # Parse and execute tool call
        tool_name, tool_arg = parse_tool_call(response)

        if tool_name and tool_name in tool_dict:
            try:
                result = await tool_dict[tool_name](tool_arg)
                if isinstance(result, list):
                    result = json.dumps(result, indent=2)
                tool_response = f"\n\nTool result:\n{result}\n\nAssistant:"
            except Exception as e:
                tool_response = f"\n\nTool error: {str(e)}\n\nAssistant:"
        else:
            # No valid tool call, treat response as final answer
            final_answer = response.strip()
            return trajectory, final_answer

        conversation = full_response + tool_response

    # Max turns reached, use last response as answer
    return trajectory, response.strip()


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
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        response=response
    )

    try:
        completion = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        judge_response = completion.choices[0].message.content.strip().lower()
        return 1.0 if "yes" in judge_response else 0.0
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0.0


def compute_reinforce_loss(
    model,
    tokenizer,
    trajectories: list,
    rewards: list,
    device: str,
    baseline: float = 0.5
) -> torch.Tensor:
    """Compute REINFORCE loss for policy gradient update.

    Uses fixed baseline of 0.5 (midpoint for binary rewards) to ensure
    gradients even when all rewards are the same.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        trajectories: List of trajectories (each is list of (prompt, response) pairs)
        rewards: List of rewards corresponding to each trajectory
        device: Device to compute on
        baseline: Fixed baseline for advantage computation

    Returns:
        Computed loss tensor
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_trajectories = 0

    for trajectory, reward in zip(trajectories, rewards):
        advantage = reward - baseline

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
            n_trajectories += 1

    return total_loss / max(n_trajectories, 1)
