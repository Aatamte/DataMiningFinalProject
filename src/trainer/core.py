"""Core Trainer class for SLM-RL training."""

import os
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.environment import load_questions, SandboxClient
from src.judge import get_local_judge_client
from src.llm_logger import LLMLogger, set_llm_logger
from src.agent import Agent, AgentConfig
from .metrics import MetricsTracker, setup_logging, setup_run_dir
from .episode import get_judge_reward, compute_reinforce_loss


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
        self.dataset = None
        self.judge_client = None
        self.logger = None
        self.metrics = None
        self.llm_logger = None
        self.device = None
        self.run_name = None
        self.run_dir = None
        self.use_wandb = False

    def setup(self) -> None:
        """Initialize all components for training."""
        cfg = self.config

        # Setup run directory and logging
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = setup_run_dir(self.run_name)
        self.logger, log_file = setup_logging(self.run_dir)
        self.metrics = MetricsTracker(self.run_dir)

        # Setup LLM call logging
        self.llm_logger = LLMLogger(self.run_dir / "logs", "llm")
        set_llm_logger(self.llm_logger)

        self._log_header(log_file)
        self._setup_wandb()

        # Load questions dataset (tools executed via Docker sandbox)
        self.logger.info("\n[1/4] Loading questions...")
        self.dataset = load_questions()

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

    def _log_header(self, log_file) -> None:
        """Log training header info."""
        cfg = self.config
        self.logger.info("=" * 60)
        self.logger.info("SLM-RL TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Run dir: {self.run_dir}")
        self.logger.info(f"Model: {cfg.model_name}")
        self.logger.info(f"Samples: {cfg.num_samples}")
        self.logger.info(f"Epochs: {cfg.num_epochs}")
        self.logger.info(f"Max turns: {cfg.max_turns}")
        self.logger.info(f"Rollouts: {cfg.num_rollouts}")
        self.logger.info(f"LR: {cfg.lr}")
        self.logger.info("=" * 60)

    def _setup_wandb(self) -> None:
        """Initialize wandb if WANDB_PROJECT env var is set."""
        wandb_project = os.environ.get("WANDB_PROJECT")
        if not wandb_project:
            self.logger.info("wandb disabled (set WANDB_PROJECT to enable)")
            return

        import wandb

        wandb_entity = os.environ.get("WANDB_ENTITY")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=self.run_name,
            config=asdict(self.config),
        )
        self.use_wandb = True
        self.logger.info(f"wandb enabled: {wandb_project}/{self.run_name}")

    def _log_wandb(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to wandb if enabled."""
        if not self.use_wandb:
            return
        import wandb
        wandb.log(metrics, step=step)

    async def train(self, live_plot: bool = False) -> None:
        """Run the training loop.

        Args:
            live_plot: If True, update plot after each step
        """
        if self.model is None:
            self.setup()

        cfg = self.config

        # Setup live plotting
        plot_func = None
        if live_plot:
            from scripts.plot_metrics import plot_metrics
            plot_path = self.run_dir / "plots.png"

            def update_plot():
                self.metrics.save()
                plot_metrics(self.metrics.metrics, output_path=plot_path, show=False)

            plot_func = update_plot
            self.logger.info(f"Live plotting enabled: {plot_path}")

        # Create subset
        subset = self.dataset.select(range(min(cfg.num_samples, len(self.dataset))))
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
                episodes = []
                rewards = []

                async with SandboxClient() as sandbox:
                    # Create agent for this sandbox session
                    agent_config = AgentConfig(
                        max_turns=cfg.max_turns,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                    agent = Agent(self.model, self.tokenizer, sandbox, agent_config)

                    for r in range(cfg.num_rollouts):
                        # Run episode using agent
                        episode = await agent.run(question)

                        # Get reward from judge
                        reward = await get_judge_reward(
                            self.judge_client, cfg.judge_model,
                            question, answer, episode.final_answer or ""
                        )

                        episodes.append(episode)
                        rewards.append(reward)

                        answer_preview = (episode.final_answer or "")[:30]
                        self.logger.info(f"    Rollout {r + 1}: reward={reward:.0f}, answer='{answer_preview}...'")

                epoch_rewards.extend(rewards)

                # Training step
                self.model.train()
                self.optimizer.zero_grad()

                loss = compute_reinforce_loss(
                    self.model, self.tokenizer, episodes, rewards, self.device
                )
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                self.logger.info(f"    Loss: {loss_val:.4f}")

                # Track metrics
                self.metrics.log_step(loss_val, rewards)
                self._log_wandb({
                    "loss": loss_val,
                    "reward_mean": sum(rewards) / len(rewards),
                    "reward_max": max(rewards),
                    "reward_min": min(rewards),
                })

                # Update live plot
                if plot_func:
                    plot_func()

                # Clear memory
                del episodes, rewards, loss
                torch.cuda.empty_cache()

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            self.metrics.log_epoch(avg_reward)
            self._log_wandb({"epoch": epoch + 1, "epoch_reward_avg": avg_reward})
            self.logger.info(f"\n  Epoch {epoch + 1} avg reward: {avg_reward:.2f}")

        # Save metrics
        metrics_file = self.metrics.save()

        # Finish wandb run
        if self.use_wandb:
            import wandb
            wandb.finish()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Metrics file: {metrics_file}")
