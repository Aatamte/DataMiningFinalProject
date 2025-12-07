"""REINFORCE training utilities."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.prompts import build_judge_prompt, parse_judge_response
from src.llm_logger import get_llm_logger
from src.agent import EpisodeResult

logger = logging.getLogger(__name__)


def group_answers_by_similarity(
    answers: list[str],
    batch_size: int = 3,
) -> list[list[int]]:
    """Group answer indices by sorting similar answers together, then chunking.

    Algorithm:
    1. Compute TF-IDF similarity matrix
    2. Greedy ordering: start with first answer, always pick most similar unvisited next
    3. Chunk the sorted indices into batches of batch_size

    This ensures similar answers are adjacent and batches are always full
    (except possibly the last one).

    Args:
        answers: List of answer strings
        batch_size: Number of answers per batch

    Returns:
        List of groups, each group is a list of answer indices.
        Example: [[0, 3, 5], [1, 4, 2], ...] - batches of similar answers
    """
    n = len(answers)

    if n <= 1:
        return [[i] for i in range(n)]

    if n <= batch_size:
        return [list(range(n))]

    # Handle empty/whitespace answers
    cleaned = [a.strip() if a else "" for a in answers]

    # TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            stop_words=None,  # Keep all words for short answers
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned)

        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # If vectorization fails (e.g., all empty), just chunk sequentially
        return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

    # Greedy ordering: start with first, pick most similar unvisited next
    visited = [False] * n
    ordered = []

    # Start with index 0
    current = 0
    ordered.append(current)
    visited[current] = True

    for _ in range(n - 1):
        # Find most similar unvisited answer to current
        best_sim = -1
        best_idx = -1
        for j in range(n):
            if not visited[j] and sim_matrix[current, j] > best_sim:
                best_sim = sim_matrix[current, j]
                best_idx = j

        if best_idx == -1:
            # Fallback: pick any unvisited
            for j in range(n):
                if not visited[j]:
                    best_idx = j
                    break

        ordered.append(best_idx)
        visited[best_idx] = True
        current = best_idx

    # Chunk into batches
    groups = []
    for i in range(0, n, batch_size):
        groups.append(ordered[i:i + batch_size])

    return groups


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
    use_approach_magnitude: bool = True,
    debug: bool = False,
    max_tokens: int = 1024,
) -> JudgeResult:
    """Get reward from judge model.

    Args:
        judge_client: OpenAI-compatible async client for judge
        judge_model: Model name to use for judging
        question: The original question
        answer: Ground truth answer
        response: Model's final answer to evaluate
        trajectory: Full conversation trajectory for approach evaluation
        use_approach_magnitude: If True, need approach score (full prompt).
                               If False, simple +1/-1 (simple prompt).
        debug: If True, print full judge prompt and response

    Returns:
        JudgeResult with reward, correctness, approach_score, and raw response
    """
    # Simple mode when we don't need approach scoring
    simple_mode = not use_approach_magnitude
    prompt = build_judge_prompt(question, answer, response, trajectory=trajectory, simple=simple_mode)
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
            max_tokens=max_tokens,  # Room for thinking + JSON response
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        judge_response = completion.choices[0].message.content

        if debug:
            print(f"[JUDGE DEBUG] RESPONSE:")
            print(f"{'='*60}")
            print(judge_response)
            print(f"{'='*60}\n")

        result = parse_judge_response(judge_response, simple=simple_mode)
        correct = result.get("correct", False)
        approach_score = result.get("approach_score", 0)

        # Clamp approach_score to 0-100
        approach_score = max(0, min(100, approach_score))

        # Reward: sign from correctness, magnitude from approach
        if correct:
            reward = approach_score / 100.0 if use_approach_magnitude else 1.0
        else:
            reward = -(1.0 - approach_score / 100.0) if use_approach_magnitude else -1.0

        if debug:
            print(f"[JUDGE DEBUG] PARSED: correct={correct}, approach_score={approach_score}, reward={reward:.3f}")

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
                max_tokens=max_tokens,
                temperature=0,
                latency_ms=latency_ms,
                parsed_result=result,
                usage=usage,
                metadata={"question": question, "answer": answer, "model_response": response},
            )

        return JudgeResult(reward=reward, correct=correct, approach_score=approach_score, raw_response=judge_response)
    except Exception as e:
        return JudgeResult(reward=-1.0, correct=False, approach_score=0, error=str(e))


def get_judge_reward_sync(
    judge_client,
    judge_model: str,
    question: str,
    answer: str,
    response: str,
    trajectory: str = "",
    use_approach_magnitude: bool = True,
    debug: bool = False,
    max_tokens: int = 1024,
) -> JudgeResult:
    """Synchronous version of get_judge_reward for thread-based execution.

    This version uses a sync OpenAI client and can run in a ThreadPoolExecutor
    for true parallelism with GPU operations (HTTP I/O releases the GIL).
    """
    # Simple mode when we don't need approach scoring
    simple_mode = not use_approach_magnitude
    prompt = build_judge_prompt(question, answer, response, trajectory=trajectory, simple=simple_mode)
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
            max_tokens=max_tokens,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        judge_response = completion.choices[0].message.content

        if debug:
            print(f"[JUDGE DEBUG] RESPONSE:")
            print(f"{'='*60}")
            print(judge_response)
            print(f"{'='*60}\n")

        result = parse_judge_response(judge_response, simple=simple_mode)
        correct = result.get("correct", False)
        approach_score = result.get("approach_score", 0)

        # Clamp approach_score to 0-100
        approach_score = max(0, min(100, approach_score))

        # Reward: sign from correctness, magnitude from approach
        if correct:
            reward = approach_score / 100.0 if use_approach_magnitude else 1.0
        else:
            reward = -(1.0 - approach_score / 100.0) if use_approach_magnitude else -1.0

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
                max_tokens=max_tokens,
                temperature=0,
                latency_ms=latency_ms,
                parsed_result=result,
                usage=usage,
                metadata={"question": question, "answer": answer, "model_response": response},
            )

        return JudgeResult(reward=reward, correct=correct, approach_score=approach_score, raw_response=judge_response)
    except Exception as e:
        return JudgeResult(reward=-1.0, correct=False, approach_score=0, error=str(e))


def compute_grpo_advantages(
    rewards: list[float],
    baseline: float = 0.5,
    debug: bool = False,
    q_idx: int = 0,
) -> list[float]:
    """Compute GRPO advantages for a group of rewards.

    Uses group mean/std as baseline. When variance is too low, falls back
    to fixed baseline to ensure learning signal.

    Args:
        rewards: List of rewards for the group
        baseline: Fallback baseline when variance is too low
        debug: If True, log diagnostic information
        q_idx: Question index for logging

    Returns:
        List of advantages corresponding to each reward
    """
    group_mean = sum(rewards) / len(rewards)
    group_var = sum((r - group_mean) ** 2 for r in rewards) / len(rewards)
    group_std = group_var ** 0.5

    # When variance is too low, fall back to REINFORCE with fixed baseline
    if group_std < 0.01:
        advantages = [r - baseline for r in rewards]
        if debug:
            print(f"[Step {q_idx}] GRPO->REINFORCE fallback: group_std={group_std:.3f}, mean={group_mean:.3f}, baseline={baseline}")
    else:
        advantages = [(r - group_mean) / (group_std + 1e-8) for r in rewards]
        if debug:
            print(f"[Step {q_idx}] GRPO: group_mean={group_mean:.3f}, group_std={group_std:.3f}")

    return advantages


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
    advantages: list[float] | None = None,
    scale_by_total_steps: int | None = None,
    skip_not_found: bool = False,
    entropy_coef: float = 0.0,
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
        advantages: Pre-computed advantages (for micro-batching). If None, computed internally.
        scale_by_total_steps: If provided, scale gradients by 1/N instead of 1/local_steps
                              (for micro-batching to maintain consistent gradient scale)
        skip_not_found: If True, skip loss computation for turns with "give up" patterns
                        (prevents model from learning to give up)
        entropy_coef: Coefficient for entropy bonus (0 = disabled). Higher values encourage
                      more diverse token distributions, preventing mode collapse.

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

    # Compute advantages if not provided (backward compatible)
    if advantages is None:
        if rl_algo == "grpo":
            advantages = compute_grpo_advantages(rewards, baseline, debug, q_idx)
        else:
            # REINFORCE: Fixed baseline
            advantages = [r - baseline for r in rewards]
            if debug:
                print(f"[Step {q_idx}] REINFORCE: baseline={baseline}")

    for ep_idx, (episode, advantage) in enumerate(zip(episodes, advantages)):
        reward = rewards[ep_idx]  # Keep for logging

        # Get trajectory in (prompt, response) format
        # Pass tokenizer to use chat template (matches generation format)
        trajectory = episode.trajectory(tokenizer)

        num_turns = len(trajectory)
        if debug:
            print(f"[Step {q_idx} R {ep_idx + 1}] reward={reward:.3f}, advantage={advantage:+.2f}, turns={num_turns}, gamma={gamma}")

        # For each turn in the trajectory, compute loss
        for turn_idx, (prompt, response) in enumerate(trajectory):
            # Skip "give up" responses (prevents learning to give up)
            # Check for common patterns that indicate the model refused to answer
            if skip_not_found:
                response_lower = response.lower()
                give_up_patterns = [
                    "not found",
                    "none found",
                    "unknown",
                    "unable to determine",
                    "unable to find",
                    "not available",
                    "cannot be determined",
                    "cannot be confirmed",
                    "cannot determine",
                    "no information",
                    "no valid",
                    "no such",
                ]
                is_give_up = any(pattern in response_lower for pattern in give_up_patterns)
                if is_give_up:
                    if debug:
                        print(f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] SKIPPED - give-up pattern in response")
                    continue

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
                    f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] "
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
                    print(f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] SKIPPED - truncation left no response tokens")
                del inputs
                continue

            # Create labels with -100 for prompt tokens (only compute loss on response)
            labels = inputs["input_ids"].clone()
            labels[0, :effective_prompt_len] = -100  # Mask prompt tokens

            # Get log probabilities (loss only on response tokens now)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Compute entropy bonus to encourage diverse outputs
            entropy_bonus = 0.0
            if entropy_coef > 0:
                # Get logits for response tokens only
                logits = outputs.logits[0, effective_prompt_len:, :]  # [response_len, vocab_size]
                # Compute entropy: H = -sum(p * log(p))
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()  # Average entropy per token
                entropy_bonus = entropy_coef * entropy

            # Weight by advantage (REINFORCE) with temporal discounting
            # positive advantage = reward > baseline = reinforce this behavior
            # negative advantage = reward < baseline = discourage this behavior
            # Earlier turns get discounted (gamma^distance_to_end), final turn gets full signal
            discount = gamma ** (num_turns - turn_idx - 1)
            discounted_advantage = advantage * discount
            weighted_loss = loss * (-discounted_advantage) - entropy_bonus  # Subtract to encourage higher entropy

            # NaN detection - skip this step if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] WARN: NaN/Inf loss detected, skipping backward()")
                del inputs, outputs, loss, weighted_loss, labels
                continue

            if debug:
                print(f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] "
                      f"prompt={effective_prompt_len} tok, response={effective_response_len} tok, total={total_len} tok | "
                      f"loss={loss.item():.4f}, discount={discount:.3f}, weighted={weighted_loss.item():.4f}")

                # Show token breakdown (prompt tokens now masked with -100)
                pct_response = (effective_response_len / total_len) * 100 if total_len > 0 else 0
                print(f"[Step {q_idx} R {ep_idx + 1} Turn {turn_idx + 1}] Loss on {effective_response_len} response tokens ({pct_response:.1f}% of sequence)")

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
    # When micro-batching, skip scaling here - caller will scale after all micro-batches
    if scale_by_total_steps is None and n_steps > 0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= n_steps

    # Log gradient accumulation diagnostic
    if debug:
        print(f"[Step {q_idx}] === GRADIENT ACCUMULATION SUMMARY ===")
        print(f"[Step {q_idx}] Total backward() calls: {n_steps}")
        if scale_by_total_steps is None:
            print(f"[Step {q_idx}] Gradients scaled by 1/{n_steps} (averaged)")
        else:
            print(f"[Step {q_idx}] Gradients NOT scaled (micro-batch mode, caller will scale)")
        print(f"[Step {q_idx}] Total prompt tokens: {diag_info['total_prompt_tokens']}, response tokens: {diag_info['total_response_tokens']}")

        # Check gradient norms
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        print(f"[Step {q_idx}] Gradient L2 norm (after scaling): {total_grad_norm:.4f}")
        print(f"[Step {q_idx}] Parameters with gradients: {param_count}")

    # Return (loss, n_steps) for micro-batching, or just loss for backward compat
    avg_loss = total_loss_value / max(n_steps, 1)
    if scale_by_total_steps is not None:
        return torch.tensor(avg_loss, device=device), n_steps
    return torch.tensor(avg_loss, device=device)
