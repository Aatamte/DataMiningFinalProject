"""Core Trainer class for SLM-RL training."""

import asyncio
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Docker-style random name generation
ADJECTIVES = [
    "happy", "clever", "brave", "calm", "eager", "fancy", "gentle", "jolly",
    "kind", "lively", "merry", "nice", "proud", "quick", "sharp", "smart",
    "swift", "warm", "wise", "bold", "bright", "cool", "fast", "keen"
]
NOUNS = [
    "panda", "falcon", "tiger", "wolf", "eagle", "hawk", "lion", "bear",
    "fox", "owl", "dolphin", "otter", "badger", "raven", "phoenix", "dragon",
    "newton", "tesla", "curie", "darwin", "fermi", "gauss", "euler", "turing"
]

def generate_run_name() -> str:
    """Generate a unique Docker-style run name."""
    return f"{random.choice(ADJECTIVES)}_{random.choice(NOUNS)}"


from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.environment import load_questions, SandboxClient
from src.judge import get_local_judge_client
from src.llm_logger import LLMLogger, set_llm_logger
from src.agent import Agent, AgentConfig
from .metrics import MetricsTracker, setup_logging, setup_run_dir
from .episode import get_judge_reward, compute_reinforce_loss
from .episode_logger import EpisodeLogger


@dataclass
class TrainingState:
    """Training state for checkpointing and resumption."""
    epoch: int = 0
    step: int = 0  # Step within current epoch
    global_step: int = 0


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_context: int = 2048

    # Training params
    num_samples: int = 50
    num_epochs: int = 3
    num_rollouts: int = 4
    max_turns: int = 3
    lr: float = 1e-5
    max_new_tokens: int = 1024
    temperature: float = 0.7  # Base temperature (used when min/max not set)
    temperature_min: float = 0.7  # Min temp for rollout sampling
    temperature_max: float = 1.15  # Max temp for rollout sampling
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    gamma: float = 0.99  # Discount factor for earlier turns in trajectory
    correctness_weight: float = 0.75  # Weight for correct answer vs approach (0-1)
    lr_scheduler: str = "linear"  # LR scheduler: "none", "linear"
    lr_end: float = 0.0  # Final LR for linear decay (default: decay to 0)
    rl_algo: str = "grpo"  # RL algorithm: "reinforce" (fixed baseline) or "grpo" (group relative)
    shuffle: bool = True  # Shuffle training samples

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    gradient_checkpointing: bool = False  # Trade compute for memory (~40% less VRAM, ~20% slower)

    # Checkpoint settings
    save_final: bool = True
    save_every_epoch: bool = False
    save_every_n_steps: int = 0  # 0 = disabled, N = save every N steps
    checkpoint_interval_minutes: int = 15  # Auto-save "latest" every N minutes (0 = disabled)
    resume_from: str = ""  # Path to checkpoint to resume from (empty = start fresh)

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
            temperature_min=get("training", "temperature_min", cls.temperature_min),
            temperature_max=get("training", "temperature_max", cls.temperature_max),
            max_grad_norm=get("training", "max_grad_norm", cls.max_grad_norm),
            gamma=get("training", "gamma", cls.gamma),
            correctness_weight=get("training", "correctness_weight", cls.correctness_weight),
            lr_scheduler=get("training", "lr_scheduler", cls.lr_scheduler),
            lr_end=get("training", "lr_end", cls.lr_end),
            rl_algo=get("training", "rl_algo", cls.rl_algo),
            shuffle=get("training", "shuffle", cls.shuffle),
            # LoRA
            use_lora=get("lora", "enabled", cls.use_lora),
            lora_r=get("lora", "r", cls.lora_r),
            lora_alpha=get("lora", "alpha", cls.lora_alpha),
            lora_dropout=get("lora", "dropout", cls.lora_dropout),
            lora_target_modules=get("lora", "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            gradient_checkpointing=get("lora", "gradient_checkpointing", cls.gradient_checkpointing),
            # Checkpoint
            save_final=get("checkpoint", "save_final", cls.save_final),
            save_every_epoch=get("checkpoint", "save_every_epoch", cls.save_every_epoch),
            save_every_n_steps=get("checkpoint", "save_every_n_steps", cls.save_every_n_steps),
            checkpoint_interval_minutes=get("checkpoint", "interval_minutes", cls.checkpoint_interval_minutes),
            resume_from=get("checkpoint", "resume_from", cls.resume_from),
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
        self.scheduler = None
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
        self._wandb_last_step = 0  # Last step logged to wandb (for resumption)
        self.training_state = TrainingState()
        self._last_checkpoint_time = 0.0  # For time-based checkpointing

    def setup(self) -> None:
        """Initialize all components for training."""
        cfg = self.config

        # Setup run directory and logging
        # Priority: RUN_ID env var > resume_from config > generate random name
        run_id = os.environ.get("RUN_ID")
        if run_id:
            self.run_name = run_id
        elif cfg.resume_from:
            self.run_name = cfg.resume_from
        else:
            self.run_name = generate_run_name()

        self.run_dir = Path(cfg.output_dir) / self.run_name
        if self.run_dir.exists():
            # Existing run - resume
            self.logger, log_file = setup_logging(self.run_dir)
            self.logger.info(f"Resuming run: {self.run_name}")
        else:
            # New run - create directory
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
        # Only use training split (first 80%) - test split reserved for eval
        self.logger.info("\n[1/4] Loading questions (train split)...")
        self.dataset = load_questions(subset="train")
        self.logger.info(f"  Train set size: {len(self.dataset)} questions")

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
                bias="none",
                target_modules=cfg.lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        else:
            self.logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Enable gradient checkpointing for memory savings
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Required for gradient checkpointing with LoRA/PEFT
            if cfg.use_lora:
                self.model.enable_input_require_grads()
            self.logger.info("  Gradient checkpointing: enabled (~40% less VRAM)")

        self.logger.info(f"  Device: {self.device}")

        # Setup optimizer (only trainable params)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr
        )

        # Setup LR scheduler
        # Note: total_steps calculated here, but scheduler created after we know num_samples
        # We'll create it in train() once we know the actual training steps
        self.scheduler = None  # Created in train()

        # Setup judge
        self.logger.info("\n[3/4] Setting up judge...")
        self.judge_client = get_local_judge_client(cfg.judge_base_url)
        self.logger.info(f"  Judge URL: {cfg.judge_base_url}")
        self.logger.info(f"  Judge model: {cfg.judge_model}")

        # Load checkpoint if resuming
        # Auto-load "latest" checkpoint if run directory exists (RUN_ID set or previous run)
        latest_checkpoint = self.run_dir / "checkpoints" / "latest"
        if cfg.resume_from:
            self.logger.info(f"\n[3.5/4] Resuming from checkpoint: {cfg.resume_from}")
            self.load_checkpoint(cfg.resume_from)
        elif latest_checkpoint.exists():
            self.logger.info(f"\n[3.5/4] Auto-resuming from: {latest_checkpoint}")
            self.load_checkpoint(str(latest_checkpoint))

        # Use wandb's last step if higher than checkpoint (handles out-of-sync cases)
        if self._wandb_last_step > self.training_state.global_step:
            self.logger.info(f"  Syncing global_step to wandb: {self.training_state.global_step} -> {self._wandb_last_step + 1}")
            self.training_state.global_step = self._wandb_last_step + 1

        # Initialize checkpoint timer
        self._last_checkpoint_time = time.time()

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
        # Use RUN_ID env var if set, otherwise use run_name
        wandb_run_id = os.environ.get("RUN_ID", self.run_name)

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=self.run_name,
            id=wandb_run_id,
            resume="allow",  # Resume if exists, create if not
            config=asdict(self.config),
        )
        self.use_wandb = True

        # Query wandb for last logged step (to continue from correct position)
        self._wandb_last_step = self._get_wandb_last_step(wandb_project, wandb_entity, wandb_run_id)
        if self._wandb_last_step > 0:
            self.logger.info(f"wandb enabled: {wandb_project}/{self.run_name} (id={wandb_run_id}, last_step={self._wandb_last_step})")
        else:
            self.logger.info(f"wandb enabled: {wandb_project}/{self.run_name} (id={wandb_run_id})")

    def _get_wandb_last_step(self, project: str, entity: str | None, run_id: str) -> int:
        """Query wandb API to get the last logged step for this run.

        Returns 0 if run doesn't exist or has no history.
        """
        try:
            import wandb
            api = wandb.Api()
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(path)
            # Get the last step from history
            history = run.history(keys=["global_step"], samples=1)
            if len(history) > 0 and "global_step" in history.columns:
                return int(history["global_step"].max())
            # Fallback: check _step which wandb uses internally
            history = run.history(keys=["_step"], samples=1)
            if len(history) > 0:
                return int(history["_step"].max())
            return 0
        except Exception:
            # Run doesn't exist yet or API error
            return 0

    def _log_wandb(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to wandb if enabled."""
        if not self.use_wandb:
            return
        import wandb
        wandb.log(metrics, step=step)

    def save_checkpoint(self, name: str = "final") -> Path:
        """Save full training checkpoint (model + optimizer + state).

        Uses atomic saving to prevent corruption if interrupted.

        Args:
            name: Checkpoint name (e.g., "final", "step_100", "latest")

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_dir = self.run_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights (LoRA adapter or full model)
        if self.config.use_lora:
            self.model.save_pretrained(checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state (optimizer, epoch, step, etc.) atomically
        state_dict = {
            "optimizer_state": self.optimizer.state_dict(),
            "training_state": asdict(self.training_state),
            "config": asdict(self.config),
            "metrics": self.metrics.metrics if self.metrics else {},
            "torch_rng_state": torch.get_rng_state(),
            "run_name": self.run_name,
        }
        if torch.cuda.is_available():
            state_dict["cuda_rng_state"] = torch.cuda.get_rng_state()

        # Atomic save: write to temp file, then replace
        state_path = checkpoint_dir / "training_state.pt"
        temp_path = checkpoint_dir / "training_state.pt.tmp"
        torch.save(state_dict, temp_path)
        temp_path.replace(state_path)  # replace() works on Windows (rename() doesn't)

        self.logger.info(f"Saved checkpoint: {checkpoint_dir} (epoch={self.training_state.epoch}, step={self.training_state.global_step})")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint directory, or just a run name
                            (e.g., "train_20251204_162911" will look for
                             runs/train_20251204_162911/checkpoints/latest)
        """
        checkpoint_dir = Path(checkpoint_path)

        # If it's just a run name, look for latest checkpoint in that run
        if not checkpoint_dir.exists():
            run_dir = Path(self.config.output_dir) / checkpoint_path / "checkpoints" / "latest"
            if run_dir.exists():
                checkpoint_dir = run_dir
            else:
                raise ValueError(f"Checkpoint not found: {checkpoint_path}\n  Tried: {checkpoint_dir} and {run_dir}")

        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            raise ValueError(f"Training state not found: {state_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_dir}")

        # Load training state
        state_dict = torch.load(state_path, map_location="cpu", weights_only=False)

        # Restore optimizer state (but reset LR to initial value)
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        # Reset LR to initial config value (scheduler will be created fresh in train())
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr

        # Restore training state
        # Only restore global_step - epoch/step reset to 0 for fresh training loop
        # This allows continued training with new epochs while preserving wandb step continuity
        ts = state_dict["training_state"]
        self.training_state = TrainingState(
            epoch=0,  # Start fresh epochs
            step=0,   # Start from first sample
            global_step=ts["global_step"],  # Continue step counter for wandb
        )

        # Restore metrics if available
        if "metrics" in state_dict and self.metrics:
            self.metrics.metrics = state_dict["metrics"]

        # Note: We intentionally do NOT restore RNG states
        # Since we reset epoch/step to 0 for fresh training, we want new randomness
        # (shuffle order, temperature sampling, etc.)

        # Load model weights
        if self.config.use_lora:
            # For LoRA, we need to load the adapter weights
            # The base model is already loaded, just load LoRA weights
            self.model.load_adapter(checkpoint_dir, adapter_name="default")
            self.logger.info("Loaded LoRA adapter weights")
        else:
            # For full model, load state dict
            self.model.load_state_dict(
                torch.load(checkpoint_dir / "pytorch_model.bin", map_location=self.device)
            )
            self.logger.info("Loaded full model weights")

        self.logger.info(f"Resumed: global_step={self.training_state.global_step} (starting fresh epochs with reset LR)")

    def _maybe_save_latest(self) -> None:
        """Save 'latest' checkpoint if enough time has passed."""
        if self.config.checkpoint_interval_minutes <= 0:
            return

        current_time = time.time()
        interval_seconds = self.config.checkpoint_interval_minutes * 60

        if current_time - self._last_checkpoint_time >= interval_seconds:
            self.save_checkpoint("latest")
            self._last_checkpoint_time = current_time

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

        # Create subset (optionally shuffled)
        num_samples = min(cfg.num_samples, len(self.dataset))
        indices = list(range(num_samples))
        if cfg.shuffle:
            # Seed with current time to ensure different order each run
            shuffle_seed = int(time.time() * 1000) % (2**32)
            random.seed(shuffle_seed)
            random.shuffle(indices)
            self.logger.info(f"  Training samples: {num_samples} (shuffled, seed={shuffle_seed})")
        else:
            self.logger.info(f"  Training samples: {num_samples} (sequential)")
        subset = self.dataset.select(indices)

        # Get starting point (for resumption)
        start_epoch = self.training_state.epoch
        start_step = self.training_state.step

        if start_epoch > 0 or start_step > 0:
            self.logger.info(f"\n[4/4] Resuming training from epoch {start_epoch + 1}, step {start_step + 1}...")
        else:
            self.logger.info(f"\n[4/4] Training for {cfg.num_epochs} epoch(s)...")

        # For ETA calculation
        total_steps = cfg.num_epochs * len(subset)
        completed_steps = start_epoch * len(subset) + start_step
        training_start_time = time.time()

        # Setup LR scheduler (now that we know total_steps)
        # Note: LR scheduler always starts fresh (from initial LR), even when resuming
        # This allows continued training with fresh LR decay while preserving global_step
        if cfg.lr_scheduler == "linear" and not cfg.eval_only:
            from torch.optim.lr_scheduler import LambdaLR
            # Calculate remaining steps from current position
            remaining_steps = total_steps - completed_steps
            # Linear decay from lr to lr_end over remaining_steps
            def lr_lambda(current_step):
                if remaining_steps == 0:
                    return 1.0
                progress = current_step / remaining_steps
                # Linear interpolation: lr * (1 - progress) + lr_end * progress
                # Simplified: lr_lambda returns multiplier, so we compute the ratio
                return (1 - progress) + (cfg.lr_end / cfg.lr) * progress
            self.scheduler = LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)  # Always start fresh
            self.logger.info(f"  LR scheduler: linear decay {cfg.lr} -> {cfg.lr_end} over {remaining_steps} remaining steps")
        else:
            self.scheduler = None
            if not cfg.eval_only:
                self.logger.info(f"  LR scheduler: none (constant lr={cfg.lr})")

        for epoch in range(start_epoch, cfg.num_epochs):
            self.logger.info(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")
            epoch_rewards = []

            # Determine starting step for this epoch
            epoch_start_step = start_step if epoch == start_epoch else 0

            for idx, item in enumerate(subset):
                # Skip already completed steps when resuming
                if idx < epoch_start_step:
                    continue

                question = item.get('question', item.get('prompt', ''))
                answer = item.get('answer', '')

                self.logger.info(f"\n  Q{idx + 1}: {question[:60]}...")
                self.logger.info(f"  Expected: {answer[:40]}...")

                # Collect rollouts for this question
                episodes = []
                rewards = []

                # Set model to eval mode for generation (disable grad checkpointing for speed)
                self.model.eval()
                if cfg.gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()

                async with SandboxClient() as sandbox:
                    # Create agent for this sandbox session
                    agent_config = AgentConfig(
                        max_turns=cfg.max_turns,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,  # Base temp (can be overridden per rollout)
                        max_context=cfg.max_context,
                        use_api=False,  # Training always uses local model
                    )
                    agent = Agent(self.model, self.tokenizer, sandbox, agent_config)

                    # Phase 1: Collect all rollouts (sequential - same sandbox)
                    for r in range(cfg.num_rollouts):
                        # Random temperature per rollout for diversity
                        temp = random.uniform(cfg.temperature_min, cfg.temperature_max)
                        episode = await agent.run(question, temperature=temp)
                        episodes.append(episode)
                        self.logger.info(f"    Rollout {r + 1} (temp={temp:.2f}): Answer = {episode.final_answer or '(none)'}")

                    # Phase 2: Judge all rollouts in parallel
                    async def judge_episode(ep):
                        if ep.final_answer is None:
                            from src.trainer.episode import JudgeResult
                            return JudgeResult(reward=0.0, correct=False, approach_score=0)
                        trajectory = ep.format_for_judge()
                        return await get_judge_reward(
                            self.judge_client, cfg.judge_model,
                            question, answer, ep.final_answer,
                            trajectory=trajectory,
                            correctness_weight=cfg.correctness_weight
                        )

                    self.logger.info("    Judging rollouts in parallel...")
                    judge_results = await asyncio.gather(*[judge_episode(ep) for ep in episodes])

                    # Phase 3: Process results and collect stats
                    correct_count = 0
                    approach_scores = []
                    answer_count = 0
                    total_turns = 0

                    for r, (episode, judge_result) in enumerate(zip(episodes, judge_results)):
                        rewards.append(judge_result.reward)
                        total_turns += episode.num_turns

                        # Track stats
                        if episode.final_answer is not None:
                            answer_count += 1
                        if judge_result.correct:
                            correct_count += 1
                        approach_scores.append(judge_result.approach_score)

                        # Detailed logging
                        if episode.final_answer is None:
                            self.logger.info(f"    Rollout {r + 1} Judge: INCORRECT (no answer)")
                        elif judge_result.error:
                            self.logger.info(f"    Rollout {r + 1} Judge error: {judge_result.error}")
                        else:
                            status = "CORRECT" if judge_result.correct else "INCORRECT"
                            self.logger.info(f"    Rollout {r + 1} Judge: {status} | Approach: {judge_result.approach_score}/100 | Reward: {judge_result.reward:.2f}")

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
                    if cfg.gradient_checkpointing:
                        self.model.gradient_checkpointing_enable()
                    self.optimizer.zero_grad()

                    # compute_reinforce_loss accumulates gradients internally via backward()
                    loss = compute_reinforce_loss(
                        self.model, self.tokenizer, episodes, rewards, self.device,
                        gamma=cfg.gamma,
                        rl_algo=cfg.rl_algo
                    )
                    # No loss.backward() needed - already done inside compute_reinforce_loss

                    # Gradient clipping for stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=cfg.max_grad_norm
                    )

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    loss_val = loss.item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"    Loss: {loss_val:.4f} | Grad norm: {grad_norm:.4f} | LR: {current_lr:.2e}")
                else:
                    self.logger.info("    [eval-only] Skipping training step")

                # Track metrics
                self.metrics.log_step(loss_val, rewards)
                n_rollouts = len(rewards)
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb_metrics = {
                    # Rewards
                    "reward_mean": sum(rewards) / n_rollouts,
                    "reward_max": max(rewards),
                    "reward_min": min(rewards),
                    # Accuracy & quality
                    "accuracy": correct_count / n_rollouts,
                    "correct_count": correct_count,
                    "approach_score_avg": sum(approach_scores) / n_rollouts,
                    "answer_rate": answer_count / n_rollouts,
                    "avg_turns": total_turns / n_rollouts,
                    # Training
                    "loss": loss_val,
                    "learning_rate": current_lr,
                    "global_step": self.training_state.global_step,
                }
                if not cfg.eval_only:
                    wandb_metrics["grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                self._log_wandb(wandb_metrics, step=self.training_state.global_step)

                # Update live plot
                if plot_func:
                    plot_func()

                # Clear memory
                del episodes, rewards
                torch.cuda.empty_cache()

                # Update training state
                self.training_state.step = idx + 1
                self.training_state.global_step += 1
                self.training_state.epoch = epoch

                # Save checkpoint if configured (by step count)
                if cfg.save_every_n_steps > 0 and self.training_state.global_step % cfg.save_every_n_steps == 0 and not cfg.eval_only:
                    self.save_checkpoint(f"step_{self.training_state.global_step}")

                # Always save "latest" checkpoint after each weight update
                if not cfg.eval_only:
                    self.save_checkpoint("latest")

                # Print progress with ETA
                completed_steps = epoch * len(subset) + idx + 1
                elapsed = time.time() - training_start_time
                steps_done_this_session = completed_steps - (start_epoch * len(subset) + start_step)
                if steps_done_this_session > 0:
                    avg_time_per_step = elapsed / steps_done_this_session
                    remaining_steps = total_steps - completed_steps
                    eta_seconds = remaining_steps * avg_time_per_step
                    eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m" if eta_seconds >= 3600 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    self.logger.info(f"  Progress: {completed_steps}/{total_steps} ({100*completed_steps/total_steps:.1f}%) | ETA: {eta_str}")

            # Reset step counter for next epoch
            self.training_state.step = 0

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
