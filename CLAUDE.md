# SLM-RL Search Project

Train Small Language Models to perform efficient programmatic search over Wikipedia using Reinforcement Learning. All components run locally (Ollama for judge, local embeddings, ChromaDB).

**Always use `uv run` to execute Python scripts in this project.**

## Quick Start

```bash
# Setup (first time only)
uv sync
ollama pull qwen2.5:7b
uv run python scripts/setup_chroma.py

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
uv run python scripts/setup_chroma.py
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Training entrypoint with CLI args |
| `scripts/setup_chroma.py` | Index Wikipedia corpus into ChromaDB |
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
python scripts/setup_chroma.py
```

**Telemetry error in Docker:**
Already fixed - `ANONYMIZED_TELEMETRY=False` is set in Dockerfile.

**Tests failing with NoneType:**
ChromaDB not loaded. Run `setup_chroma.py` and restart docker container.

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
```
