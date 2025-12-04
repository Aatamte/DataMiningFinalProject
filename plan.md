# SLM-RL Search: Local Training Plan

## Overview

Train a Small Language Model (SLM) to perform multi-turn tool-use search over Wikipedia using Reinforcement Learning. All components run locally on RTX 3090 (24GB VRAM) with zero API costs.

---

## Phase 1: Environment Setup

### 1.1 Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # Core RL/Training
    "verifiers>=0.1.7",
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "trl>=0.9.0",
    "peft>=0.11.0",
    "accelerate>=0.30.0",

    # Vector DB & Embeddings
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",

    # Data
    "datasets>=2.19.0",

    # Local LLM Inference
    "ollama>=0.2.0",

    # Utilities
    "python-dotenv>=1.0.0",
]
```

### 1.2 Install Ollama

```bash
# Windows: Download from https://ollama.com/download
# Then pull required models:
ollama pull qwen2.5:7b        # Judge model (~4.4GB download, runs in ~14GB VRAM Q4)
ollama pull qwen2.5:0.5b      # Debug SLM (~0.4GB download)
ollama pull qwen2.5:1.5b      # Larger debug option (~1GB download)
```

### 1.3 Project Structure

```
DataMining2025/
├── src/
│   ├── __init__.py
│   ├── environment.py      # Modified wiki-search env (local embeddings)
│   ├── judge.py            # Local judge wrapper (Ollama)
│   ├── embeddings.py       # Local embedding function
│   ├── training.py         # RL training loop
│   └── evaluate.py         # Evaluation scripts
├── scripts/
│   ├── setup_chroma.py     # One-time: index wiki pages
│   ├── train.py            # Main training entrypoint
│   └── eval.py             # Evaluation entrypoint
├── configs/
│   └── default.yaml        # Training hyperparameters
├── data/
│   └── .chroma_db/         # ChromaDB persistent storage
├── outputs/
│   └── checkpoints/        # Model checkpoints
├── pyproject.toml
├── README.md
├── plan.md
└── proposal.md
```

---

## Phase 2: Local Components

### 2.1 Local Embeddings

Replace OpenAI embeddings with sentence-transformers:

```python
# src/embeddings.py
from chromadb.utils import embedding_functions

def get_local_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",  # 384-dim, ~80MB, very fast
        device="cuda"
    )
```

**Considerations:**
- `all-MiniLM-L6-v2`: Fast, small, good for titles
- `all-mpnet-base-v2`: Better quality, slightly slower
- Must re-index if switching from OpenAI embeddings (different vector dimensions)

### 2.2 Local Judge

Wrap Ollama for judge inference:

```python
# src/judge.py
import ollama
from openai import AsyncOpenAI

def get_local_judge_client():
    # Ollama exposes OpenAI-compatible API on localhost:11434
    return AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # dummy key, not used
    )
```

**Judge Model Options:**
| Model | VRAM (Q4) | Quality | Speed |
|-------|-----------|---------|-------|
| qwen2.5:7b | ~14GB | Good | ~30 tok/s |
| qwen2.5:3b | ~6GB | Decent | ~50 tok/s |
| phi3:mini | ~4GB | Decent | ~60 tok/s |

Recommendation: Start with `qwen2.5:7b` for judge quality. The yes/no task is simple but we want reliable judgments.

### 2.3 Modified Environment

Create `src/environment.py` based on the original wiki-search but with:
1. Local embedding function (sentence-transformers)
2. Local judge client (Ollama)
3. Configurable paths

Key changes:
```python
def load_environment(
    # ... existing params ...
    judge_base_url: str = "http://localhost:11434/v1",  # Ollama
    embed_model: str = "all-MiniLM-L6-v2",              # Local
    use_local_embeddings: bool = True,
) -> vf.Environment:

    if use_local_embeddings:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            device="cuda"
        )
    else:
        # fallback to OpenAI if needed
        ...
```

---

## Phase 3: Data Pipeline

### 3.1 Initial Indexing

First run will:
1. Download `willcb/rare-wiki-pages` from HuggingFace
2. Index all page titles into ChromaDB with local embeddings
3. Store persistently in `data/.chroma_db/`

```bash
# One-time setup
uv run python scripts/setup_chroma.py
```

Expected time: ~5-10 minutes for ~10k pages

### 3.2 Question Dataset

`willcb/wiki-trivia-questions-v4` contains:
- Questions about the wiki pages
- Ground truth answers
- Used for both training and evaluation

We'll split:
- 80% train
- 10% validation
- 10% test

---

## Phase 4: Training Pipeline

### 4.1 Training Strategy

**Option A: GRPO (Group Relative Policy Optimization)**
- Used by verifiers library
- Sample multiple completions, rank by reward
- Update policy toward better completions

**Option B: PPO (Proximal Policy Optimization)**
- Classic RL approach via TRL
- More stable but slower

**Recommendation:** Start with GRPO via verifiers (designed for this use case)

### 4.2 Training Loop

```
For each epoch:
    For each batch of questions:
        1. Generate N rollouts per question (SLM uses tools)
        2. Score each rollout with judge
        3. Compute advantages (reward - baseline)
        4. Update SLM policy (GRPO/PPO)
        5. Log metrics
```

### 4.3 VRAM Management

RTX 3090 (24GB) budget:
- Judge inference (Ollama): ~14GB for qwen2.5:7b
- SLM training: Remaining ~10GB

**Strategy: Time-slice GPU usage**
1. Generate rollouts with SLM (unload judge)
2. Score with judge (unload SLM)
3. Train SLM (unload judge)

Or use smaller judge (qwen2.5:3b) to fit both.

### 4.4 Hyperparameters (Starting Point)

```yaml
# configs/default.yaml
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"
  max_length: 2048

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5

  # GRPO specific
  num_generations: 4        # rollouts per question
  temperature: 0.7

  # LoRA (optional, saves VRAM)
  use_lora: true
  lora_r: 16
  lora_alpha: 32

environment:
  max_turns: 10
  judge_model: "qwen2.5:7b"
  embed_model: "all-MiniLM-L6-v2"
```

---

## Phase 5: Evaluation

### 5.1 Metrics

1. **Answer Accuracy**: % of questions answered correctly (judge says "yes")
2. **Tool Efficiency**: Average number of tool calls per question
3. **Search Quality**: % of times correct page found in first search

### 5.2 Baselines

1. **Zero-shot SLM**: No RL training, just prompting
2. **Random policy**: Random tool calls
3. **Larger model**: qwen2.5:7b without training (upper bound reference)

### 5.3 Evaluation Script

```bash
uv run python scripts/eval.py \
    --model outputs/checkpoints/best \
    --split test \
    --num_samples 100
```

---

## Phase 6: Implementation Order

### Step 1: Basic Setup (Day 1)
- [ ] Add dependencies to pyproject.toml
- [ ] Run `uv sync`
- [ ] Install Ollama, pull models
- [ ] Verify Ollama works: `ollama run qwen2.5:0.5b "Hello"`

### Step 2: Local Environment (Day 1-2)
- [ ] Create `src/embeddings.py`
- [ ] Create `src/judge.py`
- [ ] Create `src/environment.py` (modified wiki-search)
- [ ] Run `scripts/setup_chroma.py` to index pages
- [ ] Test environment manually with one question

### Step 3: Training Script (Day 2-3)
- [ ] Create `configs/default.yaml`
- [ ] Create `src/training.py`
- [ ] Create `scripts/train.py`
- [ ] Debug with tiny subset (10 questions, 1 epoch)
- [ ] Verify loss decreases

### Step 4: Full Training Run (Day 3+)
- [ ] Train on full dataset
- [ ] Monitor metrics
- [ ] Save checkpoints

### Step 5: Evaluation (Day 4+)
- [ ] Create `scripts/eval.py`
- [ ] Run baselines
- [ ] Compare trained model vs baselines
- [ ] Analyze failure cases

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VRAM OOM during training | High | Use LoRA, gradient checkpointing, smaller batch |
| Judge quality too low | Medium | Try larger judge model, or improve prompt |
| Embedding mismatch | Low | Ensure consistent embedding model, re-index if needed |
| Verifiers library issues | Medium | Fall back to TRL/PPO if needed |
| Reward hacking | Medium | Monitor for degenerate behaviors, add constraints |

---

## Open Questions

1. **LoRA vs Full Fine-tuning?**
   - LoRA saves VRAM, might be enough for this task
   - Start with LoRA, try full if results plateau

2. **How many rollouts per question?**
   - More = better gradient signal, but slower
   - Start with 4, tune based on variance

3. **Judge temperature?**
   - 0 for deterministic judging? Or low temp for some variation?
   - Start with 0 (greedy) for consistent rewards

4. **Multi-turn credit assignment?**
   - How to attribute reward to individual tool calls?
   - GRPO handles this implicitly via trajectory comparison

---

## Success Criteria

**Minimum Viable:**
- Trained SLM achieves >50% accuracy on test set
- Better than zero-shot baseline

**Target:**
- >70% accuracy on test set
- <5 average tool calls per question
- Qualitative: Model learns sensible search strategies

**Stretch:**
- Approaches larger model (7B) performance
- Generalizes to out-of-distribution questions
