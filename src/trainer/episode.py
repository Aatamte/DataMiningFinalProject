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
    trajectory: str = "",
    correctness_weight: float = 0.75,
    debug: bool = False,
) -> JudgeResult:
    """Get reward from judge model.

    Reward formula: w * correct + (1-w) * (approach_score / 100)
    where w = correctness_weight

    Args:
        judge_client: OpenAI-compatible async client for judge
        judge_model: Model name to use for judging
        question: The original question
        answer: Ground truth answer
        response: Model's final answer to evaluate
        trajectory: Full conversation trajectory for approach evaluation
        correctness_weight: Weight for correctness vs approach (0-1). Default 0.75
                           means 75% weight on correct answer, 25% on approach.
        debug: If True, print full judge prompt and response

    Returns:
        JudgeResult with reward, correctness, approach_score, and raw response
    """
    prompt = build_judge_prompt(question, answer, response, trajectory=trajectory)
    messages = [{"role": "user", "content": prompt}]

    if debug:
        print(f"\n{'='*60}")
        print("[JUDGE DEBUG] PROMPT:")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")

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

        if debug:
            print(f"[JUDGE DEBUG] RESPONSE:")
            print(f"{'='*60}")
            print(judge_response)
            print(f"{'='*60}\n")

        result = parse_judge_response(judge_response)
        correct = result.get("correct", False)
        approach_score = result.get("approach_score", 0)

        if debug:
            print(f"[JUDGE DEBUG] PARSED: correct={correct}, approach_score={approach_score}, reward={correctness_weight * (1.0 if correct else 0.0) + (1-correctness_weight) * (approach_score / 100.0):.3f}")

        # Clamp approach_score to 0-100
        approach_score = max(0, min(100, approach_score))

        # Reward = w * correct + (1-w) * (approach_score / 100)
        approach_weight = 1.0 - correctness_weight
        reward = correctness_weight * (1.0 if correct else 0.0) + approach_weight * (approach_score / 100.0)

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


def get_judge_reward_sync(
    judge_client,
    judge_model: str,
    question: str,
    answer: str,
    response: str,
    trajectory: str = "",
    correctness_weight: float = 0.75,
    debug: bool = False,
) -> JudgeResult:
    """Synchronous version of get_judge_reward for thread-based execution.

    This version uses a sync OpenAI client and can run in a ThreadPoolExecutor
    for true parallelism with GPU operations (HTTP I/O releases the GIL).
    """
    prompt = build_judge_prompt(question, answer, response, trajectory=trajectory)
    messages = [{"role": "user", "content": prompt}]

    if debug:
        print(f"\n{'='*60}")
        print("[JUDGE DEBUG] PROMPT:")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")

    try:
        start_time = time.perf_counter()
        completion = judge_client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        judge_response = completion.choices[0].message.content

        if debug:
            print(f"[JUDGE DEBUG] RESPONSE:")
            print(f"{'='*60}")
            print(judge_response)
            print(f"{'='*60}\n")

        result = parse_judge_response(judge_response)
        correct = result.get("correct", False)
        approach_score = result.get("approach_score", 0)

        # Clamp approach_score to 0-100
        approach_score = max(0, min(100, approach_score))

        # Reward = w * correct + (1-w) * (approach_score / 100)
        approach_weight = 1.0 - correctness_weight
        reward = correctness_weight * (1.0 if correct else 0.0) + approach_weight * (approach_score / 100.0)

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
    gamma: float = 0.99,
    rl_algo: str = "grpo",
    debug: bool = False,
    q_idx: int = 0,
    max_context: int = 2048,
) -> torch.Tensor:
    """Compute policy gradient loss for RL training.

    Supports two algorithms:
    - "reinforce": Fixed baseline (0.5), simple policy gradient
    - "grpo": Group Relative Policy Optimization, uses group mean/std as baseline

    Uses gradient accumulation to avoid OOM - each step's gradients are
    accumulated separately rather than building one giant computation graph.

    Applies temporal discounting with gamma - earlier turns in the trajectory
    receive a discounted advantage signal, while the final turn gets full signal.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        episodes: List of EpisodeResult from agent runs
        rewards: List of rewards corresponding to each episode
        device: Device to compute on
        baseline: Fixed baseline for REINFORCE algorithm (ignored for GRPO)
        gamma: Discount factor for earlier turns (0-1). Final turn gets full
               advantage, earlier turns get gamma^(distance_to_end) discount.
        rl_algo: RL algorithm - "reinforce" (fixed baseline) or "grpo" (group relative)
        debug: If True, log detailed diagnostic information
        q_idx: Question index (1-based) for logging
        max_context: Maximum context length for truncation

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

    # Compute advantages based on RL algorithm
    if rl_algo == "grpo":
        # GRPO: Group Relative Policy Optimization
        # Use group mean as baseline, normalize by std
        group_mean = sum(rewards) / len(rewards)
        group_var = sum((r - group_mean) ** 2 for r in rewards) / len(rewards)
        group_std = group_var ** 0.5

        # When variance is too low, fall back to REINFORCE with fixed baseline
        # This ensures learning signal for both all-success and all-failure cases
        if group_std < 0.01:
            advantages = [r - baseline for r in rewards]
            if debug:
                print(f"[Q{q_idx}] GRPO->REINFORCE fallback: group_std={group_std:.3f}, mean={group_mean:.3f}, baseline={baseline}")
        else:
            advantages = [(r - group_mean) / (group_std + 1e-8) for r in rewards]
            if debug:
                print(f"[Q{q_idx}] GRPO: group_mean={group_mean:.3f}, group_std={group_std:.3f}")
    else:
        # REINFORCE: Fixed baseline
        advantages = [r - baseline for r in rewards]
        if debug:
            print(f"[Q{q_idx}] REINFORCE: baseline={baseline}")

    for ep_idx, (episode, advantage) in enumerate(zip(episodes, advantages)):
        reward = rewards[ep_idx]  # Keep for logging

        # Get trajectory in (prompt, response) format
        # Pass tokenizer to use chat template (matches generation format)
        trajectory = episode.trajectory(tokenizer)

        num_turns = len(trajectory)
        if debug:
            print(f"[Q{q_idx} Rollout {ep_idx + 1}] reward={reward:.3f}, advantage={advantage:+.2f}, turns={num_turns}, gamma={gamma}")

        # For each turn in the trajectory, compute loss
        for turn_idx, (prompt, response) in enumerate(trajectory):
            full_text = prompt + response

            # Tokenize once, measure prompt length by tokenizing prompt separately (no GPU)
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            full_len_untruncated = len(tokenizer.encode(full_text, add_special_tokens=False))

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_context
            ).to(device)

            total_len = inputs["input_ids"].shape[1]

            # Warn if truncation occurred
            if full_len_untruncated > total_len:
                logger.warning(
                    f"[Q{q_idx} Rollout {ep_idx + 1} Turn {turn_idx + 1}] "
                    f"Loss truncated: {full_len_untruncated} -> {total_len} tokens"
                )

            response_len = total_len - min(prompt_len, total_len - 1)

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
                    print(f"[Q{q_idx} Rollout {ep_idx + 1} Turn {turn_idx + 1}] SKIPPED - truncation left no response tokens")
                del inputs
                continue

            # Create labels with -100 for prompt tokens (only compute loss on response)
            labels = inputs["input_ids"].clone()
            labels[0, :effective_prompt_len] = -100  # Mask prompt tokens

            # Get log probabilities (loss only on response tokens now)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Weight by advantage (REINFORCE) with temporal discounting
            # positive advantage = reward > baseline = reinforce this behavior
            # negative advantage = reward < baseline = discourage this behavior
            # Earlier turns get discounted (gamma^distance_to_end), final turn gets full signal
            discount = gamma ** (num_turns - turn_idx - 1)
            discounted_advantage = advantage * discount
            weighted_loss = loss * (-discounted_advantage)

            # NaN detection - skip this step if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Q{q_idx} Rollout {ep_idx + 1} Turn {turn_idx + 1}] WARN: NaN/Inf loss detected, skipping backward()")
                del inputs, outputs, loss, weighted_loss, labels
                continue

            if debug:
                print(f"[Q{q_idx} Rollout {ep_idx + 1} Turn {turn_idx + 1}] "
                      f"prompt={effective_prompt_len} tok, response={effective_response_len} tok, total={total_len} tok | "
                      f"loss={loss.item():.4f}, discount={discount:.3f}, weighted={weighted_loss.item():.4f}")

                # Show token breakdown (prompt tokens now masked with -100)
                pct_response = (effective_response_len / total_len) * 100 if total_len > 0 else 0
                print(f"[Q{q_idx} Rollout {ep_idx + 1} Turn {turn_idx + 1}] Loss on {effective_response_len} response tokens ({pct_response:.1f}% of sequence)")

            # Accumulate gradients immediately (avoids building giant graph)
            # Scale by 1/n_steps will be applied after we know total steps
            weighted_loss.backward(retain_graph=False)

            total_loss_value += weighted_loss.item()
            n_steps += 1

            # Free memory
            del inputs, outputs, loss, weighted_loss, labels

    # Clear cache once at end (not per-turn - that fragments memory)
    torch.cuda.empty_cache()

    # Scale gradients by 1/n_steps (average instead of sum)
    if n_steps > 0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= n_steps

    # Log gradient accumulation diagnostic
    if debug:
        print(f"[Q{q_idx}] === GRADIENT ACCUMULATION SUMMARY ===")
        print(f"[Q{q_idx}] Total backward() calls: {n_steps}")
        print(f"[Q{q_idx}] Gradients scaled by 1/{n_steps} (averaged)")
        print(f"[Q{q_idx}] Total prompt tokens: {diag_info['total_prompt_tokens']}, response tokens: {diag_info['total_response_tokens']}")

        # Check gradient norms
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        print(f"[Q{q_idx}] Gradient L2 norm (after scaling): {total_grad_norm:.4f}")
        print(f"[Q{q_idx}] Parameters with gradients: {param_count}")

    # Return scalar tensor for logging (gradients already accumulated)
    avg_loss = total_loss_value / max(n_steps, 1)
    return torch.tensor(avg_loss, device=device)
