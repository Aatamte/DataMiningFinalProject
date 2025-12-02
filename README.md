# SLM-RL Search

Efficient programmatic search over large corpora using Small Language Models (SLMs) with Reinforcement Learning.

## Overview

This project explores using SLM-Agents to perform intelligent search over documents by combining traditional search methods with code-native logic. Instead of relying solely on vector search, TF-IDF, or keyword matching (which return many results per query), the agent learns to write Python code that filters and navigates search results using regex, conditionals, loops, and other programmatic constructs.

## Datasets

- [Wiki Trivia Questions v4](https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4)
- [Rare Wiki Pages](https://huggingface.co/datasets/willcb/rare-wiki-pages)

## Evaluation

The project uses LLM-as-a-judge within [Prime-RL environments](https://app.primeintellect.ai/dashboard/environments/will/wiki-search). Each response receives a binary good/bad score determined by comparing the SLM's predicted answer against a curated ground truth answer.

## Installation

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Install Ollama

Download and install from: https://ollama.com/download

Then pull the required models:

```bash
ollama pull qwen2.5:7b      # Judge model
ollama pull qwen2.5:0.5b    # Debug SLM
```

### 3. Index the wiki corpus

```bash
uv run python scripts/setup_chroma.py
```

This downloads the wiki pages dataset and creates local embeddings. First run takes ~5-10 minutes.

## Usage

### Test the environment

```bash
uv run python scripts/test_environment.py
```

### Train the model

```bash
uv run python scripts/train.py
```

### Evaluate

```bash
uv run python scripts/eval.py
```

## Project Structure

```
├── src/
│   ├── environment.py    # Local wiki-search environment
│   ├── embeddings.py     # Local embedding functions
│   ├── judge.py          # Local judge model wrapper
│   └── training.py       # RL training loop
├── scripts/
│   ├── setup_chroma.py   # Index wiki pages
│   ├── train.py          # Training entrypoint
│   └── eval.py           # Evaluation entrypoint
├── configs/
│   └── default.yaml      # Training hyperparameters
├── data/                  # ChromaDB storage (gitignored)
└── outputs/               # Checkpoints (gitignored)
```

## License

*TBD*

## Author

Aaron Tamte
