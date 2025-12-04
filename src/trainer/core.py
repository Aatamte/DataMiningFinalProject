"""Core Trainer class for SLM-RL training."""

import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from src.environment import load_questions, SandboxClient
from src.judge import get_local_judge_client
from src.llm_logger import LLMLogger, set_llm_logger
from src.agent import Agent, AgentConfig
from .metrics import MetricsTracker, setup_logging, setup_run_dir
from .episode import get_judge_reward, compute_reinforce_loss
from .episode_logger import EpisodeLogger


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_context: int = 2048

    # Training params
    num_samples: int = 50
    num_epochs: int = 3
    num_rollouts: int = 4
    max_turns: int = 3
    lr: float = 1e-5
    max_new_tokens: int = 1024
    temperature: float = 0.7

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Checkpoint settings
    save_final: bool = True
    save_every_epoch: bool = False

    # Judge settings
    judge_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"
    judge_base_url: str = "http://localhost:1234/v1"

    # Paths
    output_dir: str = "runs"

    # Eval mode
    eval_only: bool = False

    @classmethod
    def from_yaml(cls, config: dict) -> "TrainerConfig":
        """Create TrainerConfig from YAML config dict.

        Args:
            config: Parsed YAML configuration

        Returns:
            TrainerConfig instance
        """
        def get(section: str, key: str, default=None):
            return config.get(section, {}).get(key, default)

        return cls(
            # Model
            model_name=get("model", "name", cls.model_name),
            max_context=get("model", "max_context", cls.max_context),
            # Training
            num_samples=get("training", "num_samples", cls.num_samples),
            num_epochs=get("training", "num_epochs", cls.num_epochs),
            num_rollouts=get("training", "num_rollouts", cls.num_rollouts),
            max_turns=get("training", "max_turns", cls.max_turns),
            lr=get("training", "learning_rate", cls.lr),
            max_new_tokens=get("training", "max_new_tokens", cls.max_new_tokens),
            temperature=get("training", "temperature", cls.temperature),
            # LoRA
            use_lora=get("lora", "enabled", cls.use_lora),
            lora_r=get("lora", "r", cls.lora_r),
            lora_alpha=get("lora", "alpha", cls.lora_alpha),
            lora_dropout=get("lora", "dropout", cls.lora_dropout),
            lora_target_modules=get("lora", "target_modules", cls.lora_target_modules),
            # Checkpoint
            save_final=get("checkpoint", "save_final", cls.save_final),
            save_every_epoch=get("checkpoint", "save_every_epoch", cls.save_every_epoch),
            # Judge
            judge_model=get("judge", "model", cls.judge_model),
            judge_base_url=get("judge", "base_url", cls.judge_base_url),
            # Logging
            output_dir=get("logging", "output_dir", cls.output_dir),
        )


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
        self.episode_logger = None
        self.device = None
        self.run_name = None
        self.run_dir = None
        self.use_wandb = False

    def setup(self) -> None:
        """Initialize all components for training."""
        cfg = self.config

        # Setup run directory and logging
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = setup_run_dir(self.run_name, base_dir=cfg.output_dir)
        self.logger, log_file = setup_logging(self.run_dir)
        self.metrics = MetricsTracker(self.run_dir)

        # Setup LLM call logging
        self.llm_logger = LLMLogger(self.run_dir / "logs", "llm")
        set_llm_logger(self.llm_logger)

        # Setup episode logging
        self.episode_logger = EpisodeLogger(self.run_dir)

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

        # Apply LoRA if enabled
        if cfg.use_lora:
            self.logger.info(f"\n[2.5/4] Applying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        else:
            self.logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.logger.info(f"  Device: {self.device}")

        # Setup optimizer (only trainable params)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr
        )

        # Setup judge
        self.logger.info("\n[3/4] Setting up judge...")
        self.judge_client = get_local_judge_client(cfg.judge_base_url)
        self.logger.info(f"  Judge URL: {cfg.judge_base_url}")
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
        self.logger.info(f"LoRA: {'enabled' if cfg.use_lora else 'disabled'}")
        self.logger.info(f"Samples: {cfg.num_samples}")
        self.logger.info(f"Epochs: {cfg.num_epochs}")
        self.logger.info(f"Max turns: {cfg.max_turns}")
        self.logger.info(f"Rollouts: {cfg.num_rollouts}")
        self.logger.info(f"LR: {cfg.lr}")
        if cfg.eval_only:
            self.logger.info("Mode: EVAL ONLY (no weight updates)")
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

    def save_checkpoint(self, name: str = "final") -> Path:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (e.g., "final", "epoch_1")

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_dir = self.run_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            # Save only LoRA adapter weights
            self.model.save_pretrained(checkpoint_dir)
            self.logger.info(f"Saved LoRA adapter: {checkpoint_dir}")
        else:
            # Save full model
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            self.logger.info(f"Saved full model: {checkpoint_dir}")

        return checkpoint_dir

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
                        temperature=cfg.temperature,
                        max_context=cfg.max_context,
                        use_api=False,  # Training always uses local model
                    )
                    agent = Agent(self.model, self.tokenizer, sandbox, agent_config)

                    for r in range(cfg.num_rollouts):
                        # Run episode using agent
                        episode = await agent.run(question)

                        # Get reward from judge (skip if no answer)
                        if episode.final_answer is None:
                            # No answer = automatic 0
                            from src.trainer.episode import JudgeResult
                            judge_result = JudgeResult(reward=0.0, correct=False)
                        else:
                            judge_result = await get_judge_reward(
                                self.judge_client, cfg.judge_model,
                                question, answer, episode.final_answer
                            )

                        episodes.append(episode)
                        rewards.append(judge_result.reward)

                        # Detailed logging
                        self.logger.info(f"    Rollout {r + 1}:")
                        self.logger.info(f"      Answer: {episode.final_answer or '(none)'}")
                        if episode.final_answer is None:
                            self.logger.info("      Judge: INCORRECT (no answer)")
                        elif judge_result.error:
                            self.logger.info(f"      Judge error: {judge_result.error}")
                        else:
                            self.logger.info(f"      Judge: {'CORRECT' if judge_result.correct else 'INCORRECT'}")

                        # Log episode details
                        self.episode_logger.log_episode(
                            q_idx=idx + 1,
                            rollout_idx=r + 1,
                            question=question,
                            expected=answer,
                            episode=episode,
                            reward=judge_result.reward,
                        )

                epoch_rewards.extend(rewards)

                # Training step (skip if eval_only)
                loss_val = 0.0
                if not cfg.eval_only:
                    self.model.train()
                    self.optimizer.zero_grad()

                    # compute_reinforce_loss accumulates gradients internally via backward()
                    loss = compute_reinforce_loss(
                        self.model, self.tokenizer, episodes, rewards, self.device
                    )
                    # No loss.backward() needed - already done inside compute_reinforce_loss
                    self.optimizer.step()
                    loss_val = loss.item()
                    self.logger.info(f"    Loss: {loss_val:.4f}")
                else:
                    self.logger.info("    [eval-only] Skipping training step")

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
                del episodes, rewards
                torch.cuda.empty_cache()

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            self.metrics.log_epoch(avg_reward)
            self._log_wandb({"epoch": epoch + 1, "epoch_reward_avg": avg_reward})
            self.logger.info(f"\n  Epoch {epoch + 1} avg reward: {avg_reward:.2f}")

            # Save checkpoint after each epoch if configured
            if cfg.save_every_epoch and not cfg.eval_only:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        # Save final checkpoint
        if cfg.save_final and not cfg.eval_only:
            checkpoint_path = self.save_checkpoint("final")

        # Save metrics
        metrics_file = self.metrics.save()

        # Finish wandb run
        if self.use_wandb:
            import wandb
            wandb.finish()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Metrics: {metrics_file}")
        if cfg.save_final and not cfg.eval_only:
            self.logger.info(f"Checkpoint: {checkpoint_path}")
