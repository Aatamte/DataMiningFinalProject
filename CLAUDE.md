# SLM-RL Search Project

Train Small Language Models to perform efficient programmatic search over Wikipedia using Reinforcement Learning. All components run locally (Ollama for judge, local embeddings, ChromaDB).

**Always use `uv run` to execute Python scripts in this project.**

## Quick Start

```bash
# Setup (first time only)
uv sync
ollama pull qwen2.5:7b
uv run python scripts/setup_environment.py

# Run tests
uv run pytest tests/ -v

# Train
uv run python scripts/train.py --num_samples 10 --num_epochs 1
```

## Architecture Overview

```
Question � SLM generates tool calls � Execute tools � Judge scores answer � REINFORCE update
```

**Key Components:**
- `src/trainer/` - Training loop, REINFORCE algorithm, parsing
- `src/environment/` - Search tools, ChromaDB, sandbox client
- `src/prompts/` - System prompts for SLM and judge
- `docker/` - Isolated execution environment (optional)

## Common Commands

```bash
# Training
uv run python scripts/train.py --num_samples 50 --num_epochs 3 --num_rollouts 4

# Docker sandbox (for isolated code execution)
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml down

# Tests
uv run pytest tests/test_parsing.py -v      # Unit tests (no deps)
uv run pytest tests/test_sandbox.py -v      # Integration tests (needs docker up)

# Regenerate ChromaDB (if version mismatch errors)
rm -rf data/.chroma_db
uv run python scripts/setup_environment.py
```

## Scripts Reference

All scripts should be run with `uv run python scripts/<script>.py`.

### `setup_environment.py` - Initial Setup (Run First!)
Downloads corpus, builds ChromaDB index, and builds Docker image.

```bash
# Full setup (first time)
uv run python scripts/setup_environment.py

# Options
uv run python scripts/setup_environment.py --data-dir /path/to/data  # Custom data dir
uv run python scripts/setup_environment.py --skip-index              # Skip ChromaDB indexing
uv run python scripts/setup_environment.py --docker-only             # Only rebuild Docker image

# Dev mode (10 pages only for fast testing)
DEV=1 uv run python scripts/setup_environment.py
```

### `run_environment.py` - Start Docker Server
Stops any existing container, starts fresh, and waits for ready.

```bash
uv run python scripts/run_environment.py
# Output: Server ready! Endpoint: http://localhost:8080
```

### `test_environment.py` - Verify Environment Works
Tests all tools (search_pages, view_sections, read_section) and embedding quality. Requires server running.

```bash
uv run python scripts/run_environment.py   # Start server first
uv run python scripts/test_environment.py  # Run tests
```

### `train.py` - Training Entrypoint
Main training script with CLI arguments.

```bash
# Quick test (defaults: 3 samples, 1 epoch, 2 rollouts)
uv run python scripts/train.py

# Full training
uv run python scripts/train.py --num_samples 100 --num_epochs 3 --num_rollouts 4

# With live plotting
uv run python scripts/train.py --live-plot

# All options
uv run python scripts/train.py \
    --num_samples 50 \
    --num_epochs 3 \
    --max_turns 3 \
    --num_rollouts 4 \
    --lr 1e-5 \
    --max_new_tokens 200 \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --judge_model "qwen2.5:7b"
```

### `plot_metrics.py` - Visualize Training Results
Generates plots from a training run's metrics.json.

```bash
uv run python scripts/plot_metrics.py runs/train_20241203_143022
uv run python scripts/plot_metrics.py runs/train_20241203_143022 --output custom.png
uv run python scripts/plot_metrics.py runs/train_20241203_143022 --no-show  # Save only
```

### `test_sandbox.py` - Test Docker Sandbox (with mocks)
Tests sandbox execution with mock tool handlers (doesn't require full setup).

```bash
uv run python scripts/test_sandbox.py
```

### `run_llamacpp.py` - llama.cpp Judge Server
Runs the llama.cpp server for fast judge inference. Auto-downloads llama.cpp and model on first run.
Works on both Windows and Linux.

Reads from `.env`:
- `JUDGE_MODEL` - which model to download (e.g., `qwen3-8b`)
- `JUDGE_BASE_URL` - extracts port (e.g., port 1234 from `http://localhost:1234/v1`)

```bash
# Start server (downloads llama.cpp + model if needed)
uv run python scripts/run_llamacpp.py

# Stop server
uv run python scripts/run_llamacpp.py --stop

# Run in foreground (see logs)
uv run python scripts/run_llamacpp.py -f

# Custom settings
uv run python scripts/run_llamacpp.py --parallel 8 --ctx-size 8192
```

### `run_vllm.py` - vLLM Judge Server (Linux only)
Runs vLLM for high-throughput inference with continuous batching. Linux only (use WSL2 on Windows).
Faster than llama.cpp for batch workloads.

```bash
# Start server (installs vllm if needed)
uv run python scripts/run_vllm.py

# Stop server
uv run python scripts/run_vllm.py --stop

# Custom GPU memory (default 0.5 to leave room for training)
uv run python scripts/run_vllm.py --gpu-memory 0.7
```

Both scripts use `JUDGE_MODEL` and `JUDGE_BASE_URL` from `.env` - no config changes needed.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Training entrypoint with CLI args |
| `scripts/setup_environment.py` | Index Wikipedia corpus into ChromaDB |
| `src/trainer/core.py` | Main `Trainer` class |
| `src/trainer/episode.py` | Episode execution, REINFORCE loss |
| `src/trainer/parsing.py` | Parse tool calls and answers from LLM output |
| `src/environment/core.py` | `load_environment()` - loads everything |
| `src/environment/tools.py` | `search_pages`, `view_sections`, `read_section` |
| `docker/server.py` | FastAPI execution server (runs in container) |

## Tool Call Format

The SLM generates tool calls in this format:
```
<tool>search_pages("query")</tool>
<tool>view_sections("page_id")</tool>
<tool>read_section("section_id")</tool>
```

Final answers use:
```
<answer>The answer</answer>
```

## Training Flow

1. Sample question from dataset
2. Run N rollouts (different generations for same question)
3. Each rollout: SLM generates � parse tool/answer � execute � repeat until answer or max_turns
4. Judge (Ollama qwen2.5:7b) scores each answer: 1.0 (correct) or 0.0 (wrong)
5. Compute REINFORCE loss with baseline 0.5
6. Update model weights

## Troubleshooting

**ChromaDB `'_type'` error:**
```bash
rm -rf data/.chroma_db
python scripts/setup_environment.py
```

**Telemetry error in Docker:**
Already fixed - `ANONYMIZED_TELEMETRY=False` is set in Dockerfile.

**Tests failing with NoneType:**
ChromaDB not loaded. Run `setup_environment.py` and restart docker container.

**Ollama not responding:**
```bash
ollama serve  # Start Ollama server
ollama pull qwen2.5:7b  # Ensure model is downloaded
```

## Dependencies

- **Training:** torch, transformers, trl, peft, accelerate, verifiers
- **Vector DB:** chromadb, sentence-transformers
- **Inference:** ollama (must be installed separately)
- **Docker:** For isolated sandbox execution

## Environment Variables (.env)

```
SANDBOX_PORT=8080
DATA_DIR=./data
DOCKER_MEMORY_LIMIT=4G
DOCKER_CPU_LIMIT=2.0

# Weights & Biases (optional - set to enable logging)
WANDB_PROJECT=DataMiningSLMSearch
WANDB_ENTITY=personal-org-aaron-tamte
```
