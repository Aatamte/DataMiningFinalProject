# SLM-RL Search

Efficient programmatic search over large corpora using Small Language Models (SLMs) with Reinforcement Learning.

## Overview

This project explores using SLM-Agents to perform intelligent search over documents by combining traditional search methods with code-native logic. Instead of relying solely on vector search, TF-IDF, or keyword matching (which return many results per query), the agent learns to write Python code that filters and navigates search results using regex, conditionals, loops, and other programmatic constructs.

## Datasets

- [Wiki Trivia Questions v4](https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4)
- [Rare Wiki Pages](https://huggingface.co/datasets/willcb/rare-wiki-pages)

## Installation

### macOS (Eval only)

```bash
uv venv
uv pip install -r requirements-eval.txt
```

### Linux (Training + Eval)

```bash
uv venv
uv pip install -r requirements-training.txt
```

### Index the wiki corpus

```bash
uv run python scripts/setup_environment.py
```

This downloads the wiki pages dataset and creates local embeddings. First run takes ~5-10 minutes.

## Usage

### Evaluate

Config: `configs/eval.yaml`

```bash
uv run python scripts/eval.py
```

### Train

Config: `configs/full.yaml`

```bash
uv run python scripts/train.py
```

### Start Docker sandbox (for isolated code execution)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Project Structure

```
├── src/
│   ├── environment/      # Search environment and tools
│   ├── trainer/          # RL training loop
│   └── agent/            # Agent conversation handling
├── scripts/
│   ├── setup_environment.py  # Index wiki pages
│   ├── train.py              # Training entrypoint
│   └── eval.py               # Evaluation entrypoint
├── configs/
│   ├── full.yaml         # Training hyperparameters
│   └── eval.yaml         # Evaluation settings
├── docker/               # Sandbox execution environment
├── data/                 # ChromaDB storage (gitignored)
└── runs/                 # Checkpoints and logs (gitignored)
```

## License

*TBD*

## Author

Aaron Tamte
