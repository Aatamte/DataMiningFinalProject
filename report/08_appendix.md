## Appendix

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-4B-Instruct-2507 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj |
| Gradient checkpointing | Enabled |
| Epochs | 10 |
| Training questions | 300 |
| Rollouts per question | 20 |
| Max turns | 2 |
| Max new tokens | 256 |
| Context window | 3000 |
| Learning rate | 1e-4 |
| Max gradient norm | 1.0 |
| Temperature range | 0.7 - 1.15 |
| Discount (γ) | 0.99 |
| Correctness weight | 0.75 |
| Approach weight | 0.25 |
| RL algorithm | GRPO |
| Judge model | Qwen3-8B-AWQ |

### Training Algorithm (Pseudocode)

```
TRAIN(questions, model, judge):
    FOR each epoch:
        FOR each (question, answer) in questions:

            // Collect K rollouts with temperature diversity
            rollouts ← []
            FOR k = 1 to K:
                temp ← uniform(temp_min, temp_max)
                episode ← RUN_EPISODE(model, question, temp)
                rollouts.append(episode)

            // Judge all rollouts
            rewards ← [JUDGE(ep, answer) for ep in rollouts]

            // Compute advantages (GRPO with fallbacks)
            advantages ← COMPUTE_ADVANTAGES(rewards)
            IF advantages = skip: CONTINUE

            // Policy gradient update
            UPDATE_POLICY(model, rollouts, advantages, γ)


COMPUTE_ADVANTAGES(rewards):
    μ ← mean(rewards)
    σ ← std(rewards)

    IF σ < 0.01:
        IF μ < 0.1:
            RETURN skip                         // all failures
        ELSE:
            RETURN [r - 0.5 for r in rewards]   // REINFORCE fallback
    ELSE:
        RETURN [(r - μ) / σ for r in rewards]   // GRPO


RUN_EPISODE(model, question, temperature):
    history ← [system_prompt, question]

    FOR turn = 1 to T:
        response ← model.generate(history, temperature)

        IF response contains <answer>:
            RETURN episode with extracted answer

        IF response contains <python>:
            output ← sandbox.execute(extract_code(response))
            history.append(response, output)

    RETURN episode with no answer


JUDGE(episode, expected):
    correct ← LLM_judge.is_correct(episode.answer, expected)
    approach ← LLM_judge.score_approach(episode.trajectory)
    RETURN 0.75 × correct + 0.25 × (approach / 100)
```
