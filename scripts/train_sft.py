"""SFT training on expert trajectories.

Usage:
    uv run python scripts/train_sft.py --trajectories path/to/trajectories/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Load config
CONFIG_PATH = Path("configs/sft.yaml")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}


class TrajectoryDataset(Dataset):
    """Dataset of expert trajectories for SFT."""

    def __init__(self, trajectories_dir: Path, tokenizer, max_length: int = 3000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load all trajectory JSON files
        json_files = list(trajectories_dir.glob("*.json"))
        print(f"Found {len(json_files)} trajectory files")

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Handle both single trajectory and list of trajectories
            if isinstance(data, list):
                trajectories = data
            else:
                trajectories = [data]

            for traj in trajectories:
                messages = traj.get("messages", [])
                if messages:
                    self.examples.append(messages)

        print(f"Loaded {len(self.examples)} trajectories")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]

        # Apply chat template to get full conversation
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels = input_ids (standard causal LM training)
        # We mask padding tokens with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def setup_model(config: dict, device: str):
    """Load model with LoRA."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507")

    print(f"Loading model: {model_name}")

    # Enable TF32
    if model_cfg.get("use_tf32", True) and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "device_map": {"": 0} if device == "cuda" else None,
    }
    attention = model_cfg.get("attention", "sdpa")
    if attention != "eager" and device == "cuda":
        model_kwargs["attn_implementation"] = attention

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply LoRA
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", True):
        print(f"Applying LoRA (r={lora_cfg.get('r', 8)}, alpha={lora_cfg.get('alpha', 16)})")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = get_peft_model(model, lora_config)

        if lora_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def train(args):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup
    model, tokenizer = setup_model(CONFIG, device)

    train_cfg = CONFIG.get("training", {})
    max_context = CONFIG.get("model", {}).get("max_context", 3000)

    # Load dataset
    trajectories_dir = Path(args.trajectories)
    if not trajectories_dir.exists():
        print(f"ERROR: Trajectories directory not found: {trajectories_dir}")
        sys.exit(1)

    dataset = TrajectoryDataset(trajectories_dir, tokenizer, max_length=max_context)
    if len(dataset) == 0:
        print("ERROR: No trajectories loaded")
        sys.exit(1)

    batch_size = train_cfg.get("batch_size", 4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer & scheduler
    lr = train_cfg.get("learning_rate", 2e-5)
    num_epochs = train_cfg.get("num_epochs", 3)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Output directory
    checkpoint_cfg = CONFIG.get("checkpoint", {})
    output_dir = Path(checkpoint_cfg.get("output_dir", "runs_sft"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"sft_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SFT TRAINING")
    print(f"{'='*60}")
    print(f"Trajectories: {trajectories_dir}")
    print(f"Examples: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    # Training loop
    model.train()
    global_step = 0
    save_every = checkpoint_cfg.get("save_every_n_steps", 50)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # Save checkpoint
            if save_every > 0 and global_step % save_every == 0:
                ckpt_path = run_dir / f"checkpoint_{global_step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    # Save final model
    final_path = run_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(CONFIG, f)

    print(f"\nSFT training complete!")
    print(f"Output: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT training on expert trajectories")
    parser.add_argument("--trajectories", "-t", required=True, help="Path to trajectories directory")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
