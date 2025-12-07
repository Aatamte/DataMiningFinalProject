## Evidence of Success

### Results

Our trained agent achieves 42.68% accuracy on held-out questions, compared to TODO% for the untrained Qwen3-4B baseline—a TODO percentage point improvement from reinforcement learning. We also compare against Gemma-3-4B (4B parameters) which achieves 23.22% accuracy under the same evaluation protocol.

| Model | Parameters | Accuracy | Avg Turns |
|-------|------------|----------|-----------|
| Qwen3-4B (untrained) | 4B | TODO% | TODO |
| Qwen3-4B (trained) | 4B | 42.68% | 2.31 |
| Gemma-3-4B | 4B | 23.22% | 2.01 |

We use accuracy as our primary metric because each question has a single correct answer, making binary correctness the most direct measure of task success.

### Method

We optimize the policy using Group Relative Policy Optimization (GRPO), which computes advantages relative to the group mean and standard deviation across rollouts for each question. This provides a more stable learning signal than fixed-baseline REINFORCE when rewards are sparse. The reward function combines correctness (75%) and approach quality (25%), with a judge model (Qwen3-8B-AWQ) evaluating each trajectory.

### Configuration

Final configuration: 10 epochs, 300 questions, 20 rollouts per question, LoRA (r=8, alpha=16, dropout=0.05), temperature sampling 0.7-1.15 for trajectory diversity. Full hyperparameters are listed in the Appendix.
