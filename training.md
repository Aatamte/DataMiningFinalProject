# Training Guide

## Default Model

The default model is `Qwen/Qwen2.5-3B-Instruct`. This can be changed via:
```bash
uv run python scripts/train.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| Qwen2.5-0.5B-Instruct | ~1GB | Fast, good for testing |
| Qwen2.5-3B-Instruct | ~6GB | Better learning capacity |

## Training Duration Factors

**Per-step cost** (each question):
- `num_rollouts` episodes (default: 2)
- Each rollout: up to `max_turns` (default: 3) LLM inference + tool execution cycles
- Judge call to Ollama qwen2.5:7b after each rollout
- REINFORCE gradient update

**Total episodes** = `num_samples × num_epochs × num_rollouts`

## Time Estimates

| Configuration | Episodes | Time (GPU) | Time (CPU) |
|--------------|----------|------------|------------|
| Default (3 samples, 1 epoch, 2 rollouts) | 6 | 2-5 min | 15-30 min |
| Quick test (10 samples, 1 epoch, 2 rollouts) | 20 | 10-20 min | 1-2 hours |
| Meaningful (50 samples, 3 epochs, 4 rollouts) | 600 | 1-3 hours | 6-12 hours |
| Full (100 samples, 5 epochs, 4 rollouts) | 2000 | 4-8 hours | 20+ hours |

## Recommendations

For REINFORCE-based training to show meaningful improvement:

- **Minimum viable**: 50 samples, 3 epochs, 4 rollouts (~600 episodes)
  ```bash
  uv run python scripts/train.py --num_samples 50 --num_epochs 3 --num_rollouts 4
  ```

- **Full training**: 100+ samples, 5+ epochs
  ```bash
  uv run python scripts/train.py --num_samples 100 --num_epochs 5 --num_rollouts 4
  ```

## Bottlenecks

The main bottleneck is typically the Ollama judge calls (7B model inference) rather than the student model. Ensure Ollama is running with GPU acceleration if available:

```bash
ollama serve
```

## Monitoring

Enable live plotting to monitor training progress:
```bash
uv run python scripts/train.py --live-plot
```

Enable Weights & Biases logging by setting environment variables:
```bash
export WANDB_PROJECT=DataMiningSLMSearch
export WANDB_ENTITY=your-entity
```
