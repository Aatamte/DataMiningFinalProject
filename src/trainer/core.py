"""Core Trainer class for SLM-RL training."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
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
from src.judge import get_local_judge_client, get_sync_judge_client
from src.llm_logger import LLMLogger, set_llm_logger
from src.agent import Agent, AgentConfig
from src.prompts import build_batch_judge_prompt, parse_batch_judge_response
from .metrics import MetricsTracker, setup_logging, setup_run_dir
from .episode import get_judge_reward, get_judge_reward_sync, compute_reinforce_loss, compute_grpo_advantages, group_answers_by_similarity
from .episode_logger import EpisodeLogger


@dataclass
class TrainingState:
    """Training state for checkpointing and resumption.

    Uses a single `step` counter (global step) as source of truth.
    Epoch and step-within-epoch are derived from step and num_samples.
    """
    step: int = 0  # Global step (fetched from wandb on resume)

    def get_epoch(self, num_samples: int) -> int:
        """Get current epoch (0-indexed) from step."""
        return self.step // num_samples if num_samples > 0 else 0

    def get_step_in_epoch(self, num_samples: int) -> int:
        """Get step within current epoch (0-indexed) from step."""
        return self.step % num_samples if num_samples > 0 else 0


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_context: int = 2048
    use_tf32: bool = True  # TF32 for Ampere GPUs
    attention: str = "sdpa"  # "eager", "sdpa" (recommended), or "flash_attention_2"
    use_torch_compile: bool = False  # torch.compile for faster inference (requires build-essential)

    # Training params
    num_samples: int = 50
    num_epochs: int = 3
    num_rollouts: int = 4
    batch_splits: int = -1  # -1 = sequential, N = split rollouts into N batches for turn 1
    gradient_micro_batches: int = 1  # Split rollouts into N micro-batches for backward() - saves GPU memory
    max_turns: int = 3
    lr: float = 1e-5
    max_new_tokens: int = 1024
    temperature: float = 0.7  # Base temperature (used when min/max not set)
    temperature_min: float = 0.7  # Min temp for rollout sampling
    temperature_max: float = 1.15  # Max temp for rollout sampling
    top_p: float = 0.95  # Base top_p (nucleus sampling)
    top_p_min: float = 0.85  # Min top_p for rollout diversity
    top_p_max: float = 0.99  # Max top_p for rollout diversity
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    gamma: float = 0.99  # Discount factor for earlier turns in trajectory
    rl_algo: str = "grpo"  # RL algorithm: "reinforce" (fixed baseline) or "grpo" (group relative)

    # Reward settings
    use_approach_magnitude: bool = True  # True: approach determines magnitude, False: simple +1/-1
    skip_not_found_loss: bool = True  # Skip loss for turns with "not found" in response (prevents learning to give up)
    entropy_coef: float = 0.01  # Entropy bonus coefficient (encourages diverse outputs, prevents mode collapse)
    shuffle: bool = True  # Shuffle training samples
    async_pipeline: bool = False  # Overlap judge with next question's rollouts
    debug_loss: bool = True  # Log gradient accumulation diagnostics
    debug_judge: bool = False  # Log full judge prompts and responses
    use_8bit_adam: bool = False  # Use 8-bit Adam (halves optimizer memory, requires bitsandbytes)
    train_on_correct_only: bool = False  # Only train on episodes with correct answers (reward >= 0.75)

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    gradient_checkpointing: bool = False  # Trade compute for memory (~40% less VRAM, ~20% slower)

    # Checkpoint settings
    save_every_n_steps: int = 0  # 0 = only final, N = every N steps (also saves "latest")
    save_new_checkpoint_every: int = 0  # 0 = disabled, N = save checkpoint_{step} every N steps
    resume_from: str = ""  # Path to checkpoint to resume from (empty = start fresh)

    # Judge settings
    judge_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"
    judge_base_url: str = "http://localhost:1234/v1"
    n_batch_judge: int = 1  # Max answers per judge call (1 = individual judging)
    judge_max_tokens: int = 1024  # Max output tokens for judge response

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
            use_tf32=get("model", "use_tf32", cls.use_tf32),
            attention=get("model", "attention", cls.attention),
            use_torch_compile=get("model", "use_torch_compile", cls.use_torch_compile),
            # Training
            num_samples=get("training", "num_samples", cls.num_samples),
            num_epochs=get("training", "num_epochs", cls.num_epochs),
            num_rollouts=get("training", "num_rollouts", cls.num_rollouts),
            batch_splits=get("training", "batch_splits", cls.batch_splits),
            gradient_micro_batches=get("training", "gradient_micro_batches", cls.gradient_micro_batches),
            max_turns=get("training", "max_turns", cls.max_turns),
            lr=get("training", "learning_rate", cls.lr),
            max_new_tokens=get("training", "max_new_tokens", cls.max_new_tokens),
            temperature=get("training", "temperature", cls.temperature),
            temperature_min=get("training", "temperature_min", cls.temperature_min),
            temperature_max=get("training", "temperature_max", cls.temperature_max),
            top_p=get("training", "top_p", cls.top_p),
            top_p_min=get("training", "top_p_min", cls.top_p_min),
            top_p_max=get("training", "top_p_max", cls.top_p_max),
            max_grad_norm=get("training", "max_grad_norm", cls.max_grad_norm),
            gamma=get("training", "gamma", cls.gamma),
            rl_algo=get("training", "rl_algo", cls.rl_algo),
            # Reward
            use_approach_magnitude=get("reward", "use_approach_magnitude", cls.use_approach_magnitude),
            skip_not_found_loss=get("reward", "skip_not_found_loss", cls.skip_not_found_loss),
            entropy_coef=get("reward", "entropy_coef", cls.entropy_coef),
            shuffle=get("training", "shuffle", cls.shuffle),
            async_pipeline=get("training", "async_pipeline", cls.async_pipeline),
            debug_loss=get("training", "debug_loss", cls.debug_loss),
            debug_judge=get("training", "debug_judge", cls.debug_judge),
            use_8bit_adam=get("training", "use_8bit_adam", cls.use_8bit_adam),
            train_on_correct_only=get("training", "train_on_correct_only", cls.train_on_correct_only),
            # LoRA
            use_lora=get("lora", "enabled", cls.use_lora),
            lora_r=get("lora", "r", cls.lora_r),
            lora_alpha=get("lora", "alpha", cls.lora_alpha),
            lora_dropout=get("lora", "dropout", cls.lora_dropout),
            lora_target_modules=get("lora", "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            gradient_checkpointing=get("lora", "gradient_checkpointing", cls.gradient_checkpointing),
            # Checkpoint
            save_every_n_steps=get("checkpoint", "save_every_n_steps", cls.save_every_n_steps),
            save_new_checkpoint_every=get("checkpoint", "save_new_checkpoint_every", cls.save_new_checkpoint_every),
            resume_from=get("checkpoint", "resume_from", cls.resume_from),
            # Judge
            judge_model=get("judge", "model", cls.judge_model),
            judge_base_url=get("judge", "base_url", cls.judge_base_url),
            n_batch_judge=get("judge", "n_batch_judge", cls.n_batch_judge),
            judge_max_tokens=get("judge", "max_tokens", cls.judge_max_tokens),
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
        self.judge_client_sync = None  # Sync client for thread-based judge
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

        # Enable TF32 for Ampere GPUs (3090, A100, etc.)
        if cfg.use_tf32 and self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("  TF32 enabled for matmul and cudnn")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "device_map": {"": 0} if self.device == "cuda" else None,  # Single GPU, avoid auto overhead
        }
        if cfg.attention != "eager" and self.device == "cuda":
            model_kwargs["attn_implementation"] = cfg.attention
            self.logger.info(f"  Attention: {cfg.attention}")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

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

        # torch.compile for ~10-20% speedup (requires build-essential installed)
        if cfg.use_torch_compile and self.device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("  torch.compile: enabled (reduce-overhead mode)")

        self.logger.info(f"  Device: {self.device}")

        # Setup optimizer (only trainable params)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.Adam8bit(trainable_params, lr=cfg.lr)
                self.logger.info("  Optimizer: 8-bit Adam (bitsandbytes)")
            except ImportError:
                self.logger.warning("  bitsandbytes not installed, falling back to AdamW")
                self.optimizer = AdamW(trainable_params, lr=cfg.lr)
        else:
            self.optimizer = AdamW(trainable_params, lr=cfg.lr)

        # Setup LR scheduler
        # Note: total_steps calculated here, but scheduler created after we know num_samples
        # We'll create it in train() once we know the actual training steps
        self.scheduler = None  # Created in train()

        # Setup judge
        self.logger.info("\n[3/4] Setting up judge...")
        self.judge_client = get_local_judge_client(cfg.judge_base_url)
        self.judge_client_sync = get_sync_judge_client(cfg.judge_base_url)  # For thread-based judge
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
        if self._wandb_last_step > self.training_state.step:
            self.logger.info(f"  Syncing step to wandb: {self.training_state.step} -> {self._wandb_last_step + 1}")
            self.training_state.step = self._wandb_last_step + 1

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
        if cfg.async_pipeline:
            self.logger.info("Async pipeline: ENABLED (1-step delayed updates)")
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
        # Allow out-of-order step logging (async pipeline logs steps non-monotonically)
        wandb.define_metric("*", step_metric="step")
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
            # Get the last step from wandb's internal step counter
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

        self.logger.info(f"Saved checkpoint: {checkpoint_dir} (step={self.training_state.step})")
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
        ts = state_dict["training_state"]
        self.training_state = TrainingState(step=ts["step"])

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

        self.logger.info(f"Resumed from step {self.training_state.step} (LR reset to {self.config.lr})")
        self.training_state.step += 1  # Advance past last completed step

    async def _run_rollouts(
        self,
        agent: Agent,
        question: str,
        q_idx: int,
        cfg: TrainerConfig,
    ) -> tuple[list, list[float], float]:
        """Run all rollouts for a question.

        Args:
            agent: Agent instance with sandbox
            question: Question to answer
            q_idx: Question index (1-based for display)
            cfg: Training config

        Returns:
            Tuple of (episodes, rollout_times, total_time)
        """
        # Eval mode during generation - saves GPU memory (no gradient tracking overhead)
        # Model is set back to train() in _process_question_results before loss computation
        self.model.eval()

        episodes = []
        rollout_times = []
        all_rollouts_start = time.time()

        if cfg.batch_splits > 0:
            # Batched turn 1 generation
            rollouts_per_batch = cfg.num_rollouts // cfg.batch_splits

            # Generate random temp/top_p for each rollout upfront
            rollout_temps = [random.uniform(cfg.temperature_min, cfg.temperature_max) for _ in range(cfg.num_rollouts)]
            rollout_top_ps = [random.uniform(cfg.top_p_min, cfg.top_p_max) for _ in range(cfg.num_rollouts)]

            # Use average temp/top_p for batched turn1 generation (all in same batch)
            avg_temp = (cfg.temperature_min + cfg.temperature_max) / 2
            avg_top_p = (cfg.top_p_min + cfg.top_p_max) / 2

            # Generate turn 1 for all rollouts in batches
            turn1_results = []
            batch_start = time.time()
            for batch_idx in range(cfg.batch_splits):
                start_r = batch_idx * rollouts_per_batch
                end_r = cfg.num_rollouts if batch_idx == cfg.batch_splits - 1 else (batch_idx + 1) * rollouts_per_batch
                batch_size = end_r - start_r
                batch_results = await agent.run_batch_turn1(question, batch_size, avg_temp, avg_top_p)
                # Pair each result with its per-rollout temp/top_p
                for i, r in enumerate(batch_results):
                    rollout_idx = start_r + i
                    turn1_results.append((r, rollout_temps[rollout_idx], rollout_top_ps[rollout_idx]))
                # Yield to event loop - allows background judge tasks to make progress
                await asyncio.sleep(0)
            batch_time = time.time() - batch_start
            self.logger.info(f"[Step {q_idx}] Batch turn1 generation: {batch_time:.1f}s")

            # Continue each rollout from turn 1 with its own temp/top_p
            for r, ((conv, response, is_complete), temp, top_p) in enumerate(turn1_results):
                rollout_start = time.time()
                episode = await agent.continue_from_turn1(question, conv, response, temp, top_p)
                rollout_time = time.time() - rollout_start
                rollout_times.append(rollout_time)
                episodes.append(episode)
                avg_gen_time = rollout_time / episode.num_turns if episode.num_turns > 0 else 0
                self.logger.info(f"[Step {q_idx} R{r + 1}] (temp={temp:.2f}, top_p={top_p:.2f}): Answer = {episode.final_answer or '(none)'} [{rollout_time:.1f}s, {avg_gen_time:.1f}s/gen]")
                # Yield to event loop - allows background judge tasks to make progress
                await asyncio.sleep(0)
        else:
            # Sequential rollouts (original behavior)
            for r in range(cfg.num_rollouts):
                temp = random.uniform(cfg.temperature_min, cfg.temperature_max)
                top_p = random.uniform(cfg.top_p_min, cfg.top_p_max)
                rollout_start = time.time()
                episode = await agent.run(question, temperature=temp, top_p=top_p)
                rollout_time = time.time() - rollout_start
                rollout_times.append(rollout_time)
                episodes.append(episode)
                avg_gen_time = rollout_time / episode.num_turns if episode.num_turns > 0 else 0
                self.logger.info(f"[Step {q_idx} R{r + 1}] (temp={temp:.2f}, top_p={top_p:.2f}): Answer = {episode.final_answer or '(none)'} [{rollout_time:.1f}s, {avg_gen_time:.1f}s/gen]")
                # Yield to event loop - allows background judge tasks to make progress
                await asyncio.sleep(0)

        total_rollout_time = time.time() - all_rollouts_start
        avg_rollout_time = sum(rollout_times) / len(rollout_times) if rollout_times else 0
        self.logger.info(f"[Step {q_idx}] Rollouts complete: {total_rollout_time:.1f}s total, {avg_rollout_time:.1f}s avg/rollout")

        return episodes, rollout_times, total_rollout_time

    async def _judge_episodes(
        self,
        episodes: list,
        question: str,
        answer: str,
        q_idx: int,
        cfg: TrainerConfig,
    ) -> tuple[list, list[float], int, list[int], int, int]:
        """Judge all episodes and compute rewards.

        Supports two modes:
        - n_batch_judge == 1: Individual judging (parallel via ThreadPoolExecutor)
        - n_batch_judge > 1: Batch judging (group similar answers, fewer API calls)

        Args:
            episodes: List of EpisodeResult
            question: Original question
            answer: Expected answer
            q_idx: Question index (1-based for display)
            cfg: Training config

        Returns:
            Tuple of (judge_results, rewards, correct_count, approach_scores, answer_count, total_turns)
        """
        from src.trainer.episode import JudgeResult

        judge_start = time.time()

        # Initialize results array (indexed by episode)
        judge_results = [None] * len(episodes)

        if cfg.n_batch_judge > 1:
            # === BATCH JUDGING MODE ===
            # Group similar answers and judge in batches

            # Extract answers for similarity grouping
            answers_list = [
                ep.final_answer if ep.final_answer is not None else "(no answer)"
                for ep in episodes
            ]

            # Sort by similarity, then chunk into batches
            groups = group_answers_by_similarity(
                answers_list,
                batch_size=cfg.n_batch_judge,
            )

            if cfg.debug_judge:
                self.logger.info(f"[Step {q_idx}] Batch judge: {len(episodes)} episodes -> {len(groups)} groups")

            def judge_batch_sync(group_indices: list[int]) -> list[dict]:
                """Judge a batch of episodes synchronously."""
                # Build entries for batch prompt
                entries = []
                for idx in group_indices:
                    ep = episodes[idx]
                    entries.append({
                        "id": idx,
                        "response": ep.final_answer if ep.final_answer is not None else "(no answer)",
                        "trajectory": ep.format_for_judge(),
                    })

                # Build and send batch prompt (simple mode when not using approach magnitude)
                simple_mode = not cfg.use_approach_magnitude
                prompt = build_batch_judge_prompt(question, answer, entries, simple=simple_mode)
                messages = [{"role": "user", "content": prompt}]

                try:
                    completion = self.judge_client_sync.chat.completions.create(
                        model=cfg.judge_model,
                        messages=messages,
                        max_tokens=cfg.judge_max_tokens,
                        temperature=0,
                    )
                    response_text = completion.choices[0].message.content
                    results = parse_batch_judge_response(response_text, simple=simple_mode)
                    return results
                except Exception as e:
                    # On error, return wrong results for all in batch
                    self.logger.warning(f"[Step {q_idx}] Batch judge error: {e}")
                    return [{"id": idx, "correct": False, "approach_score": 0, "error": str(e)} for idx in group_indices]

            # Run batch judge calls in parallel (one per group)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(groups)) as executor:
                futures = [
                    loop.run_in_executor(executor, judge_batch_sync, group)
                    for group in groups
                ]
                batch_results = await asyncio.gather(*futures)

            # Map batch results back to episodes
            for batch_result in batch_results:
                for item in batch_result:
                    idx = item["id"]
                    ep = episodes[idx]

                    # Compute reward from correctness and approach score
                    correct = item["correct"]
                    approach_score = item["approach_score"]

                    # Reward: sign from correctness, magnitude from approach
                    if correct and ep.final_answer is not None:
                        if cfg.use_approach_magnitude:
                            reward = approach_score / 100.0  # +0.0 to +1.0
                        else:
                            reward = 1.0
                    else:
                        if cfg.use_approach_magnitude:
                            reward = -(1.0 - approach_score / 100.0)  # -1.0 to -0.0
                        else:
                            reward = -1.0
                        correct = False

                    judge_results[idx] = JudgeResult(
                        reward=reward,
                        correct=correct,
                        approach_score=approach_score,
                    )

            # Fill any missing results with defaults (treat as wrong answer)
            for idx in range(len(episodes)):
                if judge_results[idx] is None:
                    judge_results[idx] = JudgeResult(reward=-1.0, correct=False, approach_score=0, error="Missing from batch")

        else:
            # === INDIVIDUAL JUDGING MODE ===
            def judge_episode_sync(ep, ep_idx):
                """Synchronous judge call - runs in thread, HTTP I/O releases GIL."""
                trajectory = ep.format_for_judge()
                response = ep.final_answer if ep.final_answer is not None else "(no answer)"

                result = get_judge_reward_sync(
                    self.judge_client_sync, cfg.judge_model,
                    question, answer, response,
                    trajectory=trajectory,
                    use_approach_magnitude=cfg.use_approach_magnitude,
                    debug=cfg.debug_judge,
                    max_tokens=cfg.judge_max_tokens,
                )

                # Reward: sign from correctness, magnitude from approach
                if result.correct and ep.final_answer is not None:
                    if cfg.use_approach_magnitude:
                        reward = result.approach_score / 100.0  # +0.0 to +1.0
                    else:
                        reward = 1.0
                else:
                    if cfg.use_approach_magnitude:
                        reward = -(1.0 - result.approach_score / 100.0)  # -1.0 to -0.0
                    else:
                        reward = -1.0

                return JudgeResult(
                    reward=reward,
                    correct=result.correct if ep.final_answer is not None else False,
                    approach_score=result.approach_score,
                    raw_response=result.raw_response,
                )

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(episodes)) as executor:
                futures = [
                    loop.run_in_executor(executor, judge_episode_sync, ep, i)
                    for i, ep in enumerate(episodes)
                ]
                judge_results = await asyncio.gather(*futures)

        judge_time = time.time() - judge_start

        # Process results
        rewards = []
        correct_count = 0
        approach_scores = []
        answer_count = 0
        total_turns = 0

        for r, (episode, judge_result) in enumerate(zip(episodes, judge_results)):
            rewards.append(judge_result.reward)
            total_turns += episode.num_turns

            if episode.final_answer is not None:
                answer_count += 1
            if judge_result.correct:
                correct_count += 1
            approach_scores.append(judge_result.approach_score)

            # Detailed logging
            if episode.final_answer is None:
                self.logger.info(f"[Step {q_idx} R{r + 1}] Judge: NO ANSWER (penalty) | Approach: {judge_result.approach_score}/100 | Reward: {judge_result.reward:.2f}")
            elif judge_result.error:
                self.logger.info(f"[Step {q_idx} R{r + 1}] Judge error: {judge_result.error}")
            else:
                status = "CORRECT" if judge_result.correct else "INCORRECT"
                self.logger.info(f"[Step {q_idx} R{r + 1}] Judge: {status} | Approach: {judge_result.approach_score}/100 | Reward: {judge_result.reward:.2f}")

        self.logger.info(f"[Step {q_idx}] Judge complete: {judge_time:.1f}s")

        return judge_results, rewards, correct_count, approach_scores, answer_count, total_turns

    def _apply_update(
        self,
        episodes: list,
        rewards: list[float],
        correct_count: int,
        approach_scores: list[int],
        answer_count: int,
        total_turns: int,
        idx: int,
        question: str,
        answer: str,
        cfg: TrainerConfig,
        step: int | None = None,
    ) -> tuple[float, float | None]:
        """Apply training update for a batch of episodes.

        Args:
            episodes: List of EpisodeResult
            rewards: Rewards for each episode
            correct_count: Number of correct answers
            approach_scores: Approach scores for each episode
            answer_count: Number of episodes with answers
            total_turns: Total turns across all episodes
            idx: Question index (for episode logging)
            question: Original question
            answer: Expected answer
            cfg: Training config
            step: Global step for logging (if None, uses idx + 1)

        Returns:
            Tuple of (loss_value, grad_norm)
        """
        # Use step for logging, fall back to idx + 1 for backward compat
        log_step = step if step is not None else idx + 1

        # Log episodes
        for r, (episode, reward) in enumerate(zip(episodes, rewards)):
            self.episode_logger.log_episode(
                q_idx=log_step,
                rollout_idx=r + 1,
                question=question,
                expected=answer,
                episode=episode,
                reward=reward,
            )

        loss_val = 0.0
        grad_norm = None

        if not cfg.eval_only:
            import time as _time
            _t_start = _time.time()

            self.model.train()
            if cfg.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            self.optimizer.zero_grad()
            _t_setup = _time.time()

            # Filter to correct-only if enabled
            if cfg.train_on_correct_only:
                correct_threshold = 0.75  # Correct answers have reward >= 0.75
                original_count = len(episodes)
                filtered = [(ep, r) for ep, r in zip(episodes, rewards) if r >= correct_threshold]
                if not filtered:
                    self.logger.info(f"[Step {log_step}] No correct episodes (0/{original_count}), skipping update")
                    return 0.0, None
                episodes, rewards = zip(*filtered)
                episodes, rewards = list(episodes), list(rewards)
                self.logger.info(f"[Step {log_step}] Training on {len(episodes)}/{original_count} correct episodes")

            # Compute advantages once for ALL episodes (for consistent GRPO baseline)
            if cfg.rl_algo == "grpo":
                all_advantages = compute_grpo_advantages(
                    rewards, baseline=0.5, debug=cfg.debug_loss, q_idx=log_step
                )
            else:
                all_advantages = [r - 0.5 for r in rewards]  # REINFORCE baseline

            # Micro-batching: split episodes into smaller groups for backward()
            n_micro = cfg.gradient_micro_batches
            _t_advantages = _time.time()
            if n_micro > 1 and len(episodes) > n_micro:
                # Split into micro-batches
                micro_batch_size = len(episodes) // n_micro
                total_n_steps = 0
                total_loss = 0.0
                self.logger.info(f"[Step {log_step}] Starting {n_micro} micro-batches...")

                for mb_idx in range(n_micro):
                    _t_mb_start = _time.time()
                    start_i = mb_idx * micro_batch_size
                    # Last micro-batch gets remainder
                    end_i = len(episodes) if mb_idx == n_micro - 1 else (mb_idx + 1) * micro_batch_size

                    mb_episodes = episodes[start_i:end_i]
                    mb_rewards = rewards[start_i:end_i]
                    mb_advantages = all_advantages[start_i:end_i]

                    # compute_reinforce_loss returns (loss, n_steps) when scale_by_total_steps is set
                    loss, n_steps = compute_reinforce_loss(
                        self.model, self.tokenizer, mb_episodes, mb_rewards, self.device,
                        gamma=cfg.gamma,
                        rl_algo=cfg.rl_algo,
                        q_idx=log_step,
                        max_context=cfg.max_context,
                        debug=cfg.debug_loss and mb_idx == 0,  # Only debug first micro-batch
                        advantages=mb_advantages,
                        scale_by_total_steps=True,  # Don't scale inside, we'll scale after
                        skip_not_found=cfg.skip_not_found_loss,
                        entropy_coef=cfg.entropy_coef,
                    )
                    total_n_steps += n_steps
                    total_loss += loss.item() * n_steps
                    _t_mb_end = _time.time()

                    self.logger.info(f"[Step {log_step}] Micro-batch {mb_idx + 1}/{n_micro}: {len(mb_episodes)} episodes, {n_steps} steps, {_t_mb_end - _t_mb_start:.2f}s")

                # Scale gradients by total steps across all micro-batches
                if total_n_steps > 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= total_n_steps

                loss_val = total_loss / max(total_n_steps, 1)
            else:
                # No micro-batching (or only 1 micro-batch) - use original flow
                loss = compute_reinforce_loss(
                    self.model, self.tokenizer, episodes, rewards, self.device,
                    gamma=cfg.gamma,
                    rl_algo=cfg.rl_algo,
                    q_idx=log_step,
                    max_context=cfg.max_context,
                    debug=cfg.debug_loss,
                    advantages=all_advantages,  # Still use pre-computed advantages for consistency
                    skip_not_found=cfg.skip_not_found_loss,
                    entropy_coef=cfg.entropy_coef,
                )
                loss_val = loss.item()

            _t_loss = _time.time()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=cfg.max_grad_norm
            )
            _t_clip = _time.time()

            self.optimizer.step()
            _t_optim = _time.time()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Timing breakdown
            self.logger.info(
                f"[Step {log_step}] Gradient update: setup={_t_setup - _t_start:.2f}s, "
                f"loss+backward={_t_loss - _t_setup:.2f}s, clip={_t_clip - _t_loss:.2f}s, "
                f"optim={_t_optim - _t_clip:.2f}s, total={_t_optim - _t_start:.2f}s"
            )
            self.logger.info(f"[Step {log_step}] Loss: {loss_val:.4f} | Grad norm: {grad_norm:.4f} | LR: {current_lr:.2e}")
        else:
            self.logger.info(f"[Step {log_step}] [eval-only] Skipping training step")

        return loss_val, grad_norm

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
        # Shuffle ALL training indices first, then take num_samples
        # This ensures we sample randomly from the entire training set, not just first N
        num_samples = min(cfg.num_samples, len(self.dataset))
        all_indices = list(range(len(self.dataset)))
        if cfg.shuffle:
            # Seed with current time to ensure different order each run
            shuffle_seed = int(time.time() * 1000) % (2**32)
            random.seed(shuffle_seed)
            random.shuffle(all_indices)
            indices = all_indices[:num_samples]
            self.logger.info(f"  Training samples: {num_samples} / {len(self.dataset)} (random sample, seed={shuffle_seed})")
        else:
            indices = all_indices[:num_samples]
            self.logger.info(f"  Training samples: {num_samples} (sequential, first {num_samples})")
        subset = self.dataset.select(indices)

        # Get starting point (for resumption) - step is the single source of truth
        start_step = self.training_state.step
        num_samples = len(subset)
        total_steps = cfg.num_epochs * num_samples

        if start_step > 0:
            start_epoch = start_step // num_samples
            start_step_in_epoch = start_step % num_samples
            self.logger.info(f"\n[4/4] Resuming from step {start_step} (epoch {start_epoch + 1}, sample {start_step_in_epoch + 1})...")
        else:
            start_epoch = 0
            start_step_in_epoch = 0
            self.logger.info(f"\n[4/4] Training for {cfg.num_epochs} epoch(s), {total_steps} total steps...")

        # For ETA calculation
        start_global_step = start_step  # Remember where we started this session
        training_start_time = time.time()

        # Constant LR (no scheduler)
        self.scheduler = None
        if not cfg.eval_only:
            self.logger.info(f"  Learning rate: {cfg.lr}")

        for epoch in range(start_epoch, cfg.num_epochs):
            self.logger.info(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")
            if cfg.async_pipeline:
                self.logger.info("  [Async pipeline enabled - overlapping judge with next rollouts]")
            epoch_rewards = []

            # Determine starting sample for this epoch
            epoch_start_sample = start_step_in_epoch if epoch == start_epoch else 0

            # For async pipeline: track pending update from previous question
            pending_update = None  # Dict with episodes, question, answer, idx, step, judge_task

            # Helper to process a completed question (judge results -> update -> metrics -> checkpoint)
            async def _process_question_results(
                pending: dict,
                epoch_rewards: list,
                epoch: int,
                plot_func,
            ):
                """Process judge results and apply update for a question."""
                # Wait for judge results
                judge_start = time.time()
                _, rewards, correct_count, approach_scores, answer_count, total_turns = await pending['judge_task']
                judge_time = time.time() - judge_start

                episodes = pending['episodes']
                question = pending['question']
                answer = pending['answer']
                idx = pending['idx']
                step = pending['step']  # Global step for this question (set when queued)

                epoch_rewards.extend(rewards)

                # Apply training update (pass step for logging)
                loss_val, grad_norm = self._apply_update(
                    episodes, rewards, correct_count, approach_scores,
                    answer_count, total_turns, idx, question, answer, cfg,
                    step=step
                )

                # Track metrics
                self.metrics.log_step(loss_val, rewards)
                n_rollouts = len(rewards)
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb_metrics = {
                    "reward_mean": sum(rewards) / n_rollouts,
                    "reward_max": max(rewards),
                    "reward_min": min(rewards),
                    "accuracy": correct_count / n_rollouts,
                    "correct_count": correct_count,
                    "approach_score_avg": sum(approach_scores) / n_rollouts,
                    "answer_rate": answer_count / n_rollouts,
                    "avg_turns": total_turns / n_rollouts,
                    "loss": loss_val,
                    "learning_rate": current_lr,
                    "step": step,
                }
                if not cfg.eval_only and grad_norm is not None:
                    wandb_metrics["grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm

                # Update training state
                self.training_state.step = step

                self._log_wandb(wandb_metrics, step=step)

                if plot_func:
                    plot_func()

                # Clear memory (empty_cache called in compute_reinforce_loss)
                del episodes, rewards

                # Save checkpoint if configured
                if cfg.save_every_n_steps > 0 and step % cfg.save_every_n_steps == 0 and not cfg.eval_only:
                    self.save_checkpoint("latest")

                # Save numbered checkpoint (preserves history)
                if cfg.save_new_checkpoint_every > 0 and step % cfg.save_new_checkpoint_every == 0 and not cfg.eval_only:
                    self.save_checkpoint(f"checkpoint_{step}")

                # Print progress with ETA
                elapsed = time.time() - training_start_time
                steps_done_this_session = step - start_global_step
                if steps_done_this_session > 0:
                    avg_time_per_step = elapsed / steps_done_this_session
                    remaining_steps = total_steps - step
                    eta_seconds = remaining_steps * avg_time_per_step
                    eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m" if eta_seconds >= 3600 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    self.logger.info(f"[Step {step}] Progress: {step}/{total_steps} ({100*step/total_steps:.1f}%) | ETA: {eta_str}")

            for idx, item in enumerate(subset):
                # Skip already completed samples when resuming
                if idx < epoch_start_sample:
                    continue

                # Calculate current step (global step for this sample)
                step = epoch * num_samples + idx + 1

                question = item.get('question', item.get('prompt', ''))
                answer = item.get('answer', '')
                is_last_question = (idx == num_samples - 1)

                timestamp = datetime.now().strftime("%H:%M:%S")
                self.logger.info(f"\n  [{timestamp}] Step {step}: {question[:60]}...")
                self.logger.info(f"  Expected: {answer[:40]}...")

                # Set model to eval mode for generation
                self.model.eval()
                if cfg.gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()

                async with SandboxClient() as sandbox:
                    agent_config = AgentConfig(
                        max_turns=cfg.max_turns,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        max_context=cfg.max_context,
                        use_api=False,
                    )
                    agent = Agent(self.model, self.tokenizer, sandbox, agent_config)

                    # Run rollouts for current question
                    episodes, rollout_times, total_rollout_time = await self._run_rollouts(agent, question, step, cfg)

                    if cfg.async_pipeline and pending_update is not None:
                        # Process previous question's results (judge was running in background)
                        await _process_question_results(pending_update, epoch_rewards, epoch, plot_func)
                        pending_update = None

                    if cfg.async_pipeline and not is_last_question:
                        # Start judge in background, continue to next question
                        self.logger.info(f"[Step {step}] Starting judge in background...")
                        judge_task = asyncio.create_task(
                            self._judge_episodes(episodes, question, answer, step, cfg)
                        )
                        # Yield immediately to let judge task start (send HTTP requests)
                        await asyncio.sleep(0)
                        pending_update = {
                            'episodes': episodes,
                            'question': question,
                            'answer': answer,
                            'idx': idx,
                            'step': step,
                            'judge_task': judge_task,
                        }
                    else:
                        # Sync mode or last question: judge and update now
                        self.logger.info(f"[Step {step}] Judging rollouts...")
                        _, rewards, correct_count, approach_scores, answer_count, total_turns = await self._judge_episodes(
                            episodes, question, answer, step, cfg
                        )
                        epoch_rewards.extend(rewards)

                        # Apply training update
                        loss_val, grad_norm = self._apply_update(
                            episodes, rewards, correct_count, approach_scores,
                            answer_count, total_turns, idx, question, answer, cfg,
                            step=step
                        )

                        # Track metrics
                        self.metrics.log_step(loss_val, rewards)
                        n_rollouts = len(rewards)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        wandb_metrics = {
                            "reward_mean": sum(rewards) / n_rollouts,
                            "reward_max": max(rewards),
                            "reward_min": min(rewards),
                            "accuracy": correct_count / n_rollouts,
                            "correct_count": correct_count,
                            "approach_score_avg": sum(approach_scores) / n_rollouts,
                            "answer_rate": answer_count / n_rollouts,
                            "avg_turns": total_turns / n_rollouts,
                            "loss": loss_val,
                            "learning_rate": current_lr,
                            "step": step,
                        }
                        if not cfg.eval_only and grad_norm is not None:
                            wandb_metrics["grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm

                        # Update training state
                        self.training_state.step = step

                        self._log_wandb(wandb_metrics, step=step)

                        if plot_func:
                            plot_func()

                        # Clear memory (empty_cache called in compute_reinforce_loss)
                        del episodes, rewards

                        # Save checkpoint if configured
                        if cfg.save_every_n_steps > 0 and step % cfg.save_every_n_steps == 0 and not cfg.eval_only:
                            self.save_checkpoint("latest")

                        # Save numbered checkpoint (preserves history)
                        if cfg.save_new_checkpoint_every > 0 and step % cfg.save_new_checkpoint_every == 0 and not cfg.eval_only:
                            self.save_checkpoint(f"checkpoint_{step}")

                        # Print progress with ETA
                        elapsed = time.time() - training_start_time
                        steps_done_this_session = step - start_global_step
                        if steps_done_this_session > 0:
                            avg_time_per_step = elapsed / steps_done_this_session
                            remaining_steps = total_steps - step
                            eta_seconds = remaining_steps * avg_time_per_step
                            eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m" if eta_seconds >= 3600 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                            self.logger.info(f"[Step {step}] Progress: {step}/{total_steps} ({100*step/total_steps:.1f}%) | ETA: {eta_str}")

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            self.metrics.log_epoch(avg_reward)
            # Log epoch metrics with last step of the epoch to keep wandb steps monotonic
            last_step_of_epoch = (epoch + 1) * num_samples
            self._log_wandb({"epoch": epoch + 1, "epoch_reward_avg": avg_reward}, step=last_step_of_epoch)
            self.logger.info(f"\n  Epoch {epoch + 1} avg reward: {avg_reward:.2f}")

        # Save final checkpoint
        if not cfg.eval_only:
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
        if not cfg.eval_only:
            self.logger.info(f"Checkpoint: {checkpoint_path}")
