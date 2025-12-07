## Conclusions

This project demonstrates that a 4-billion parameter model can learn effective information-seeking behavior through reinforcement learning alone, without human demonstrations. Our trained agent achieves TODO% accuracy on held-out questions, a TODO percentage point improvement over the untrained baseline, while learning non-trivial strategies like query formulation, hierarchical navigation, and programmatic filtering of search results. The model's ability to compose Python code that adapts to retrieved content—rather than following fixed retrieval patterns—represents a step toward more flexible information-seeking agents.

Future work could explore two directions: (1) scaling to larger base models (8B+) to test whether increased capacity enables more complex multi-hop reasoning, and (2) training the model to express uncertainty and reformulate queries when initial retrieval fails, rather than hallucinating answers.
