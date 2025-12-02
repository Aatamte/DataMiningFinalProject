"""Minimal end-to-end training script.

Start with tiny params to verify everything works:
    uv run python scripts/train.py

Then scale up:
    uv run python scripts/train.py --num_samples 100 --num_epochs 3
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, ".")

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.environment import load_environment
from src.judge import get_local_judge_client
from src.prompts import SYSTEM_PROMPT, JUDGE_PROMPT, build_prompt


def setup_logging(output_dir: str, run_name: str) -> logging.Logger:
    """Set up logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"{run_name}.log")

    # Create logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler (more detailed)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


class MetricsTracker:
    """Track training metrics."""

    def __init__(self, output_dir: str, run_name: str):
        self.output_dir = output_dir
        self.run_name = run_name
        self.metrics = {
            "steps": [],
            "losses": [],
            "rewards": [],
            "epoch_avg_rewards": [],
        }
        self.step = 0

    def log_step(self, loss: float, rewards: list):
        """Log metrics for a training step."""
        self.step += 1
        self.metrics["steps"].append(self.step)
        self.metrics["losses"].append(loss)
        self.metrics["rewards"].append(sum(rewards) / len(rewards) if rewards else 0)

    def log_epoch(self, avg_reward: float):
        """Log epoch-level metrics."""
        self.metrics["epoch_avg_rewards"].append(avg_reward)

    def save(self):
        """Save metrics to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_file = os.path.join(self.output_dir, f"{self.run_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return metrics_file


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLM on wiki-search")

    # Minimal defaults for testing
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of questions to train on (default: 3)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Max tool-use turns per episode (default: 3)")
    parser.add_argument("--num_rollouts", type=int, default=2,
                        help="Rollouts per question (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Max new tokens per generation (default: 200)")

    # Model settings
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model to train")
    parser.add_argument("--judge_model", type=str, default="qwen2.5:7b",
                        help="Ollama model for judging")

    # Paths
    parser.add_argument("--chroma_db_dir", type=str, default="data/.chroma_db",
                        help="ChromaDB directory")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints",
                        help="Output directory for checkpoints")

    return parser.parse_args()


def build_tool_prompt(tools):
    """Build the tool description prompt."""
    tool_descs = []
    for tool in tools:
        doc = tool.__doc__ or ""
        tool_descs.append(f"- {tool.__name__}: {doc.split(chr(10))[0]}")
    return "\n".join(tool_descs)


def parse_tool_call(text):
    """Parse a tool call from model output.

    Expected format: <tool>tool_name(arg1, arg2)</tool>
    Or: tool_name("arg")
    """
    # Try XML-style format
    match = re.search(r'<tool>(\w+)\(([^)]*)\)</tool>', text)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        return func_name, args_str

    # Try function-style format
    match = re.search(r'(\w+)\("([^"]*)"\)', text)
    if match:
        return match.group(1), match.group(2)

    # Try simple format: search_pages: query
    match = re.search(r'(search_pages|view_sections|read_section)[:\s]+["\']?([^"\'<\n]+)["\']?', text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).strip()

    return None, None


async def run_episode(model, tokenizer, question, answer, tools, tool_dict, max_turns, max_new_tokens, device, max_context=1500):
    """Run a single episode: model uses tools to answer a question."""
    conversation = build_prompt(question)
    initial_len = len(conversation)
    trajectory = []  # Store (prompt, response) pairs

    for turn in range(max_turns):
        # Truncate conversation if too long (keep system prompt + recent context)
        if len(conversation) > max_context * 4:  # rough char estimate
            # Keep initial prompt and last portion
            conversation = conversation[:initial_len] + "\n...(truncated)...\n" + conversation[-(max_context * 2):]

        # Generate response
        inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=max_context).to(device)

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
        answer_match = re.search(r'<answer>([^<]+)</answer>', response)
        if answer_match:
            final_answer = answer_match.group(1).strip()
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


async def get_judge_reward(judge_client, judge_model, question, answer, response):
    """Get reward from judge model."""
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


def compute_reinforce_loss(model, tokenizer, trajectories, rewards, device, baseline=0.5):
    """Compute REINFORCE loss for policy gradient update.

    Uses fixed baseline of 0.5 (midpoint for binary rewards) to ensure
    gradients even when all rewards are the same.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_trajectories = 0

    for trajectory, reward in zip(trajectories, rewards):
        advantage = reward - baseline

        # For each turn in the trajectory, compute loss
        for prompt, response in trajectory:
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

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


async def main_async():
    args = parse_args()

    # Setup logging
    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, log_file = setup_logging(args.output_dir, run_name)
    metrics = MetricsTracker(args.output_dir, run_name)

    logger.info("=" * 60)
    logger.info("SLM-RL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Run: {run_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Rollouts: {args.num_rollouts}")
    logger.info(f"LR: {args.lr}")
    logger.info("=" * 60)

    # Load environment
    logger.info("\n[1/4] Loading environment...")
    env = load_environment(
        max_turns=args.max_turns,
        judge_model=args.judge_model,
        chroma_db_dir=args.chroma_db_dir,
    )

    # Get tools
    tools = env.tools
    tool_dict = {t.__name__: t for t in tools}
    logger.info(f"  Tools: {list(tool_dict.keys())}")

    # Create subset
    subset = env.dataset.select(range(min(args.num_samples, len(env.dataset))))
    logger.info(f"  Training samples: {len(subset)}")

    # Load model
    logger.info(f"\n[2/4] Loading model: {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    logger.info(f"  Device: {device}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Setup judge
    logger.info("\n[3/4] Setting up judge...")
    judge_client = get_local_judge_client()
    logger.info(f"  Judge model: {args.judge_model}")

    # Training loop
    logger.info(f"\n[4/4] Training for {args.num_epochs} epoch(s)...")

    for epoch in range(args.num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        epoch_rewards = []

        for idx, item in enumerate(subset):
            question = item.get('question', item.get('prompt', ''))
            answer = item.get('answer', '')

            logger.info(f"\n  Q{idx + 1}: {question[:60]}...")
            logger.info(f"  Expected: {answer[:40]}...")

            # Collect rollouts for this question
            rollouts = []
            rewards = []

            for r in range(args.num_rollouts):
                # Run episode
                trajectory, final_answer = await run_episode(
                    model, tokenizer, question, answer, tools, tool_dict,
                    args.max_turns, args.max_new_tokens, device
                )

                # Get reward from judge
                reward = await get_judge_reward(
                    judge_client, args.judge_model,
                    question, answer, final_answer
                )

                rollouts.append(trajectory)
                rewards.append(reward)

                logger.info(f"    Rollout {r + 1}: reward={reward:.0f}, answer='{final_answer[:30]}...'")

            epoch_rewards.extend(rewards)

            # Training step
            model.train()
            optimizer.zero_grad()

            loss = compute_reinforce_loss(model, tokenizer, rollouts, rewards, device)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            logger.info(f"    Loss: {loss_val:.4f}")

            # Track metrics
            metrics.log_step(loss_val, rewards)

            # Clear memory
            del rollouts, rewards, loss
            torch.cuda.empty_cache()

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        metrics.log_epoch(avg_reward)
        logger.info(f"\n  Epoch {epoch + 1} avg reward: {avg_reward:.2f}")

    # Save metrics
    metrics_file = metrics.save()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Metrics file: {metrics_file}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
