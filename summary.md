# SLM-RL Search: Project Summary

## Project Overview

This project trains Small Language Models (SLMs) to perform **programmatic search over Wikipedia** using Reinforcement Learning. Instead of returning a ranked list of documents, the model learns to search like a human researcher: formulating targeted queries, scanning results, drilling into relevant sections, and synthesizing answers.

**Core insight**: We combine three capabilities that are individually well-studied but rarely integrated:
1. Code generation (model writes Python to call search tools)
2. Tool use with real execution (code runs in Docker sandbox)
3. Reinforcement learning from sparse task rewards (GRPO/REINFORCE)

The model learns to *write code that searches*, not just predict the next token.

---

## Architecture

```
Question → SLM generates <python>...</python> → Execute in sandbox →
→ Observe results → Repeat (up to 3 turns) → <answer>...</answer> →
→ Judge scores (correct? approach quality?) → REINFORCE/GRPO update
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Agent** (`src/agent/`) | Manages conversation, generates code/answers, executes in sandbox |
| **Trainer** (`src/trainer/core.py`) | Training loop, checkpointing, wandb logging |
| **Episode** (`src/trainer/episode.py`) | REINFORCE/GRPO loss computation, judge integration |
| **Judge** (`src/prompts/judge.py`) | LLM-as-judge for correctness + approach scoring |
| **Sandbox** (`docker/`) | Isolated code execution with search tools |

---

## Training Configuration

### Model
- **Base model**: `Qwen/Qwen3-4B-Instruct-2507` (4B parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank: 16, Alpha: 32, Dropout: 0.05
  - Target modules: Q, K, V, O projections
  - Trainable params: ~0.1% of total

### RL Algorithm Options

| Algorithm | Description | When to Use |
|-----------|-------------|-------------|
| **GRPO** (default) | Group Relative Policy Optimization - uses rollout group mean/std as baseline | Better for diverse rewards |
| **REINFORCE** | Fixed baseline (0.5) | Simpler, works when rewards bounded [0,1] |

### Reward Function

```
R = w × correct + (1-w) × (approach_score / 100)
```

- `w = 0.75` (correctness weight)
- `correct`: 1.0 if answer matches ground truth, else 0.0
- `approach_score`: 0-100 rating of search strategy quality

**Why this split?** Prevents reward hacking (can't just guess randomly) and rewards good search strategies that fail to find the answer (partial credit for good process).

### Temporal Discounting

Earlier turns in a trajectory receive discounted advantage signal:
- Final turn: full signal (γ⁰ = 1.0)
- Turn before: γ¹ = 0.99
- Two turns before: γ² = 0.98

This helps with credit assignment in multi-turn episodes.

### Hyperparameters

| Parameter | Default | Full | Small | Debug |
|-----------|---------|------|-------|-------|
| `num_samples` | 50 | 10 | 10 | 1 |
| `num_epochs` | 3 | 3 | 2 | 1 |
| `num_rollouts` | 4 | 4 | 3 | 2 |
| `max_turns` | 3 | 3 | 3 | 3 |
| `learning_rate` | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| `max_new_tokens` | 1024 | 512 | 512 | 512 |
| `temperature` | 0.7 | 0.7 | 0.7 | 0.7 |
| `temperature_min` | 0.7 | 0.7 | 0.7 | 0.7 |
| `temperature_max` | 1.15 | 1.15 | 1.15 | 1.15 |
| `gamma` | 0.99 | 0.99 | 0.99 | 0.99 |
| `correctness_weight` | 0.75 | 0.75 | 0.75 | 0.75 |
| `lr_scheduler` | linear | linear | linear | linear |
| `rl_algo` | grpo | grpo | grpo | grpo |
| `max_grad_norm` | 1.0 | 1.0 | - | - |

### Temperature Diversity

Each rollout samples a random temperature from `[temp_min, temp_max]` to encourage exploration diversity. This helps GRPO have meaningful variance to learn from.

---

## Judge Model

- **Model**: DeepSeek-R1-0528-Qwen3-8B (via LM Studio/Ollama)
- **Endpoint**: OpenAI-compatible API at `localhost:1234`

### Scoring Criteria

**Correctness** (true/false):
- True only if final answer contains ground truth or semantic equivalent
- False if wrong, missing, "not found", or just code

**Approach Score** (0-100):
- 0-25: Relevant search queries (targeted vs generic)
- 0-25: Efficiency (minimal wasted turns)
- 0-25: Found and read appropriate sources
- 0-25: Clear reasoning, answer from evidence

---

## Config Profiles

| Profile | Purpose | Samples × Epochs × Rollouts | Est. Time |
|---------|---------|----------------------------|-----------|
| `debug.yaml` | Quick validation | 1 × 1 × 2 | ~2 min |
| `small.yaml` | Testing/iteration | 10 × 2 × 3 | ~30 min |
| `full.yaml` | Production training | 10 × 3 × 4 | ~1 hour |
| `default.yaml` | Full training | 50 × 3 × 4 | ~5 hours |

---

## Recent Updates

1. **GRPO Algorithm**: Added Group Relative Policy Optimization as default RL algorithm
2. **Temperature Diversity**: Rollouts use random temperature in [0.7, 1.15] range
3. **Temporal Discounting**: γ=0.99 discount for earlier turns in trajectory
4. **Correctness Weight**: 75/25 split between correctness and approach quality
5. **Linear LR Scheduler**: Decays learning rate to 0 over training
6. **Gradient Clipping**: max_grad_norm=1.0 for training stability
7. **Shuffle**: Training samples shuffled each epoch

---

## Running Training

```bash
# Quick debug run
uv run python scripts/train.py --config configs/debug.yaml

# Small test run
uv run python scripts/train.py --config configs/small.yaml

# Full training
uv run python scripts/train.py --config configs/full.yaml

# With live plotting
uv run python scripts/train.py --config configs/small.yaml --live-plot
```

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/*.yaml` | Training configurations |
| `src/trainer/core.py` | Main Trainer class, training loop |
| `src/trainer/episode.py` | REINFORCE/GRPO loss, judge integration |
| `src/agent/agent.py` | Agent for running episodes |
| `src/prompts/judge.py` | Judge prompt and parsing |
| `scripts/train.py` | CLI entrypoint |

---

## Metrics Tracked

- `reward_mean/max/min`: Reward statistics per step
- `accuracy`: Fraction of rollouts with correct answers
- `approach_score_avg`: Average approach quality
- `answer_rate`: Fraction of rollouts that produced an answer
- `avg_turns`: Average turns per episode
- `loss`: Policy gradient loss
- `grad_norm`: Gradient norm (for stability monitoring)
- `learning_rate`: Current LR (for scheduler tracking)
