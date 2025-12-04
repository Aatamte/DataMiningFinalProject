# Evaluation Guide

How to evaluate models on the wiki-search task.

## Quick Start

```bash
# Evaluate base model (1% of questions, 1 run each)
uv run python scripts/eval.py

# Evaluate 10% of questions
uv run python scripts/eval.py --q_percentage 10

# Multiple runs per question (for variance estimation)
uv run python scripts/eval.py --q_percentage 5 --n_per_q 3

# Evaluate a fine-tuned checkpoint
uv run python scripts/eval.py --model outputs/checkpoints/my_model
```

## Prerequisites

1. **Judge model running** - LM Studio or Ollama serving the judge model
2. **Docker sandbox running** - `uv run python scripts/run_environment.py`

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `TRAIN_MODEL` env | HuggingFace model name or LoRA checkpoint path |
| `--base_model` | None | Base model for LoRA checkpoints (required if model is LoRA) |
| `--q_percentage` | 1.0 | Percentage of total questions to evaluate (0-100) |
| `--n_per_q` | 1 | Number of evaluation runs per question |
| `--max_turns` | 3 | Max tool-use turns per question |
| `--max_new_tokens` | 1024 | Max tokens per generation |
| `--judge_model` | `JUDGE_MODEL` env | Model for judging answers |

## Output

Results are always saved to `evals/` directory:

```
evals/
└── eval_Qwen_Qwen3-4B_20241204_143022/
    └── results.json
```

### Results JSON Structure

```json
{
  "model": "Qwen/Qwen3-4B",
  "q_percentage": 5.0,
  "n_per_q": 3,
  "total_questions": 10000,
  "num_questions": 500,
  "total_runs": 1500,
  "accuracy": 62.5,
  "avg_turns": 2.1,
  "results": [
    {
      "question": "What is...",
      "expected": "Ground truth",
      "runs": [
        {"run": 1, "answer": "...", "correct": true, "num_turns": 2},
        {"run": 2, "answer": "...", "correct": false, "num_turns": 3},
        {"run": 3, "answer": "...", "correct": true, "num_turns": 2}
      ],
      "accuracy": 66.7
    }
  ]
}
```

## Comparing Models

### Base vs Fine-tuned (LoRA)

```bash
# Evaluate base model
uv run python scripts/eval.py --model Qwen/Qwen2.5-0.5B-Instruct --q_percentage 10

# Evaluate LoRA fine-tuned checkpoint
uv run python scripts/eval.py \
    --model runs/train_20241204_143022/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --q_percentage 10
```

**Note:** For LoRA checkpoints, you must specify `--base_model` with the original model that was fine-tuned.

### Variance Estimation

Use `--n_per_q` to run multiple evals per question and measure consistency:

```bash
# 3 runs per question to measure variance
uv run python scripts/eval.py --q_percentage 5 --n_per_q 3
```

Per-question accuracy shows how consistently the model answers each question.

## Metrics

- **Accuracy** - % of runs answered correctly (total_correct / total_runs)
- **Avg turns** - Average tool-use turns per run (lower = more efficient)
- **Per-question accuracy** - % correct across n_per_q runs for each question

## Interpreting Results

| Accuracy | Interpretation |
|----------|----------------|
| < 20% | Model not learning / broken |
| 20-40% | Baseline (untrained model) |
| 40-60% | Some learning |
| 60-80% | Good performance |
| > 80% | Excellent |

## Troubleshooting

**"Cannot connect to sandbox"**
```bash
uv run python scripts/run_environment.py
```

**"Judge model not found"**
- Check LM Studio/Ollama is running
- Verify model name matches: `--judge_model <model_id>`

**Low accuracy despite training**
- Check episode logs in `runs/*/episodes/` for failure patterns
- Model may need more training samples or epochs

---

## Results

Baseline evaluations on wiki-search task.

| Model | Quant | Accuracy | Avg Turns | Questions | Notes |
|-------|-------|----------|-----------|-----------|-------|
| Qwen3-4B-Instruct-2507 | F16 | | | | |
| google/gemma-3-4b | | | | | |
| Qwen3-1.7B | Q8_0 | | | | |
