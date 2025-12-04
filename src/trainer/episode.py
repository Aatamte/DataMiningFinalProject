"""REINFORCE training utilities."""

import logging
import time
from dataclasses import dataclass

import torch

from src.prompts import build_judge_prompt, parse_judge_response
from src.llm_logger import get_llm_logger
from src.agent import EpisodeResult

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from judge evaluation."""
    reward: float
    correct: bool
    approach_score: int = 0  # 0-100 score for approach quality
    raw_response: str | None = None
    error: str | None = None


async def get_judge_reward(
    judge_client,
    judge_model: str,
    question: str,
    answer: str,
    response: str,
    trajectory: str = ""
) -> JudgeResult:
    """Get reward from judge model.

    Reward formula: 0.5 * correct + 0.5 * (approach_score / 100)
    - Correct answer contributes up to 0.5
    - Good approach contributes up to 0.5
    - Total range: 0.0 to 1.0

    Args:
        judge_client: OpenAI-compatible async client for judge
        judge_model: Model name to use for judging
        question: The original question
        answer: Ground truth answer
        response: Model's final answer to evaluate
        trajectory: Full conversation trajectory for approach evaluation

    Returns:
        JudgeResult with reward, correctness, approach_score, and raw response
    """
    prompt = build_judge_prompt(question, answer, response, trajectory=trajectory)
    messages = [{"role": "user", "content": prompt}]

    try:
        start_time = time.perf_counter()
        completion = await judge_client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=4096,  # R1 models need room for <think> block + JSON
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        judge_response = completion.choices[0].message.content
        result = parse_judge_response(judge_response)
        correct = result.get("correct", False)
        approach_score = result.get("approach_score", 0)

        # Clamp approach_score to 0-100
        approach_score = max(0, min(100, approach_score))

        # Reward = 0.5 * correct + 0.5 * (approach_score / 100)
        reward = 0.5 * (1.0 if correct else 0.0) + 0.5 * (approach_score / 100.0)

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
                max_tokens=4096,
                temperature=0,
                latency_ms=latency_ms,
                parsed_result=result,
                usage=usage,
                metadata={"question": question, "answer": answer, "model_response": response},
            )

        return JudgeResult(reward=reward, correct=correct, approach_score=approach_score, raw_response=judge_response)
    except Exception as e:
        return JudgeResult(reward=0.0, correct=False, approach_score=0, error=str(e))


def compute_reinforce_loss(
    model,
    tokenizer,
    episodes: list[EpisodeResult],
    rewards: list[float],
    device: str,
    baseline: float = 0.5,
    debug: bool = True
) -> torch.Tensor:
    """Compute REINFORCE loss for policy gradient update.

    Uses fixed baseline of 0.5 (midpoint for binary rewards) to ensure
    gradients even when all rewards are the same.

    Uses gradient accumulation to avoid OOM - each step's gradients are
    accumulated separately rather than building one giant computation graph.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        episodes: List of EpisodeResult from agent runs
        rewards: List of rewards corresponding to each episode
        device: Device to compute on
        baseline: Fixed baseline for advantage computation
        debug: If True, log detailed diagnostic information

    Returns:
        Computed loss tensor (scalar for logging, gradients already accumulated)
    """
    total_loss_value = 0.0
    n_steps = 0

    # Diagnostic tracking
    diag_info = {
        "total_prompt_tokens": 0,
        "total_response_tokens": 0,
        "total_tokens": 0,
    }

    for ep_idx, (episode, reward) in enumerate(zip(episodes, rewards)):
        advantage = reward - baseline

        # Get trajectory in (prompt, response) format
        # Pass tokenizer to use chat template (matches generation format)
        trajectory = episode.trajectory(tokenizer)

        if debug:
            print(f"  [DIAG] Episode {ep_idx}: reward={reward}, advantage={advantage:+.2f}, turns={len(trajectory)}")

        # For each turn in the trajectory, compute loss
        for turn_idx, (prompt, response) in enumerate(trajectory):
            full_text = prompt + response

            # Tokenize separately to measure lengths
            prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            response_tokens = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"]

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)

            prompt_len = prompt_tokens.shape[1]
            response_len = response_tokens.shape[1]
            total_len = inputs["input_ids"].shape[1]

            # Handle truncation: if prompt+response was truncated, adjust prompt_len
            # Ensure at least 1 response token is unmasked
            effective_prompt_len = min(prompt_len, total_len - 1)
            effective_response_len = total_len - effective_prompt_len

            # Track for diagnostics (use effective lengths)
            diag_info["total_prompt_tokens"] += effective_prompt_len
            diag_info["total_response_tokens"] += effective_response_len
            diag_info["total_tokens"] += total_len

            # Skip if truncation left no response tokens
            if effective_response_len <= 0:
                if debug:
                    print(f"    [DIAG] Turn {turn_idx}: SKIPPED - truncation left no response tokens")
                del inputs, prompt_tokens, response_tokens
                continue

            # Create labels with -100 for prompt tokens (only compute loss on response)
            labels = inputs["input_ids"].clone()
            labels[0, :effective_prompt_len] = -100  # Mask prompt tokens

            # Get log probabilities (loss only on response tokens now)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Weight by advantage (REINFORCE)
            # positive advantage = reward > baseline = reinforce this behavior
            # negative advantage = reward < baseline = discourage this behavior
            weighted_loss = loss * (-advantage)

            # NaN detection - skip this step if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    [WARN] Turn {turn_idx}: NaN/Inf loss detected, skipping backward()")
                del inputs, outputs, loss, weighted_loss, prompt_tokens, response_tokens, labels
                torch.cuda.empty_cache()
                continue

            if debug:
                print(f"    [DIAG] Turn {turn_idx}: "
                      f"prompt={effective_prompt_len} tok, response={effective_response_len} tok, total={total_len} tok | "
                      f"loss={loss.item():.4f}, weighted={weighted_loss.item():.4f}")

                # Show token breakdown (prompt tokens now masked with -100)
                pct_response = (effective_response_len / total_len) * 100 if total_len > 0 else 0
                print(f"    [DIAG] Loss computed on {effective_response_len} response tokens ({pct_response:.1f}% of sequence)")

            # Accumulate gradients immediately (avoids building giant graph)
            # Scale by 1/n_steps will be applied after we know total steps
            weighted_loss.backward(retain_graph=False)

            total_loss_value += weighted_loss.item()
            n_steps += 1

            # Free memory
            del inputs, outputs, loss, weighted_loss, prompt_tokens, response_tokens, labels
            torch.cuda.empty_cache()

    # Scale gradients by 1/n_steps (average instead of sum)
    if n_steps > 0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= n_steps

    # Log gradient accumulation diagnostic
    if debug:
        print(f"  [DIAG] === GRADIENT ACCUMULATION SUMMARY ===")
        print(f"  [DIAG] Total backward() calls: {n_steps}")
        print(f"  [DIAG] Gradients scaled by 1/{n_steps} (averaged)")
        print(f"  [DIAG] Total prompt tokens in loss: {diag_info['total_prompt_tokens']}")
        print(f"  [DIAG] Total response tokens in loss: {diag_info['total_response_tokens']}")
        print(f"  [DIAG] Prompt/Response ratio: {diag_info['total_prompt_tokens']}/{diag_info['total_response_tokens']}")

        # Check gradient norms
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  [DIAG] Gradient L2 norm (after scaling): {total_grad_norm:.4f}")
        print(f"  [DIAG] Parameters with gradients: {param_count}")

    # Return scalar tensor for logging (gradients already accumulated)
    avg_loss = total_loss_value / max(n_steps, 1)
    return torch.tensor(avg_loss, device=device)
