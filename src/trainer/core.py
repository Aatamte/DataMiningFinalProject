"""Core Trainer class for SLM-RL training."""

from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.environment import load_environment
from src.judge import get_local_judge_client
from .metrics import MetricsTracker, setup_logging
from .episode import run_episode, get_judge_reward, compute_reinforce_loss


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Training params
    num_samples: int = 3
    num_epochs: int = 1
    max_turns: int = 3
    num_rollouts: int = 2
    lr: float = 1e-5
    max_new_tokens: int = 200

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    judge_model: str = "qwen2.5:7b"

    # Paths
    chroma_db_dir: str = "data/.chroma_db"
    output_dir: str = "outputs/checkpoints"


class Trainer:
    """Main trainer class for SLM-RL Search."""

    def __init__(self, config: TrainerConfig | None = None, **kwargs):
        """Initialize the trainer.

        Args:
            config: TrainerConfig instance. If None, creates default config.
            **kwargs: Override config values (e.g., num_samples=100)
        """
        if config is None:
            config = TrainerConfig()

        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.env = None
        self.judge_client = None
        self.logger = None
        self.metrics = None
        self.device = None
        self.run_name = None

    def setup(self) -> None:
        """Initialize all components for training."""
        cfg = self.config

        # Setup logging
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger, log_file = setup_logging(cfg.output_dir, self.run_name)
        self.metrics = MetricsTracker(cfg.output_dir, self.run_name)

        self._log_header(log_file)

        # Load environment
        self.logger.info("\n[1/4] Loading environment...")
        self.env = load_environment(
            max_turns=cfg.max_turns,
            judge_model=cfg.judge_model,
            chroma_db_dir=cfg.chroma_db_dir,
        )
        self.logger.info(f"  Tools: {[t.__name__ for t in self.env.tools]}")

        # Load model
        self.logger.info(f"\n[2/4] Loading model: {cfg.model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr)

        # Setup judge
        self.logger.info("\n[3/4] Setting up judge...")
        self.judge_client = get_local_judge_client()
        self.logger.info(f"  Judge model: {cfg.judge_model}")

    def _log_header(self, log_file: str) -> None:
        """Log training header info."""
        cfg = self.config
        self.logger.info("=" * 60)
        self.logger.info("SLM-RL TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Model: {cfg.model_name}")
        self.logger.info(f"Samples: {cfg.num_samples}")
        self.logger.info(f"Epochs: {cfg.num_epochs}")
        self.logger.info(f"Max turns: {cfg.max_turns}")
        self.logger.info(f"Rollouts: {cfg.num_rollouts}")
        self.logger.info(f"LR: {cfg.lr}")
        self.logger.info("=" * 60)

    async def train(self) -> None:
        """Run the training loop."""
        if self.model is None:
            self.setup()

        cfg = self.config

        # Get tools
        tools = self.env.tools
        tool_dict = {t.__name__: t for t in tools}

        # Create subset
        subset = self.env.dataset.select(range(min(cfg.num_samples, len(self.env.dataset))))
        self.logger.info(f"  Training samples: {len(subset)}")

        self.logger.info(f"\n[4/4] Training for {cfg.num_epochs} epoch(s)...")

        for epoch in range(cfg.num_epochs):
            self.logger.info(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")
            epoch_rewards = []

            for idx, item in enumerate(subset):
                question = item.get('question', item.get('prompt', ''))
                answer = item.get('answer', '')

                self.logger.info(f"\n  Q{idx + 1}: {question[:60]}...")
                self.logger.info(f"  Expected: {answer[:40]}...")

                # Collect rollouts for this question
                rollouts = []
                rewards = []

                for r in range(cfg.num_rollouts):
                    # Run episode
                    trajectory, final_answer = await run_episode(
                        self.model, self.tokenizer, question, answer,
                        tools, tool_dict, cfg.max_turns, cfg.max_new_tokens, self.device
                    )

                    # Get reward from judge
                    reward = await get_judge_reward(
                        self.judge_client, cfg.judge_model,
                        question, answer, final_answer
                    )

                    rollouts.append(trajectory)
                    rewards.append(reward)

                    self.logger.info(f"    Rollout {r + 1}: reward={reward:.0f}, answer='{final_answer[:30]}...'")

                epoch_rewards.extend(rewards)

                # Training step
                self.model.train()
                self.optimizer.zero_grad()

                loss = compute_reinforce_loss(
                    self.model, self.tokenizer, rollouts, rewards, self.device
                )
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                self.logger.info(f"    Loss: {loss_val:.4f}")

                # Track metrics
                self.metrics.log_step(loss_val, rewards)

                # Clear memory
                del rollouts, rewards, loss
                torch.cuda.empty_cache()

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            self.metrics.log_epoch(avg_reward)
            self.logger.info(f"\n  Epoch {epoch + 1} avg reward: {avg_reward:.2f}")

        # Save metrics
        metrics_file = self.metrics.save()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Metrics file: {metrics_file}")
