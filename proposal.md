Aaron Tamte
Repo Link: https://github.com/2025-F-CS6220/project-slm-rl-search

Datasets: https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4
https://huggingface.co/datasets/willcb/rare-wiki-pages


The problem I am trying to solve is efficient programmatic search over a large corpus of data/content using small language models (SLM). For instance, instead of relying on traditional search methods like vector search, TF-IDF, keyword, etc where one query might return 10+ results, I want to give a SLM-Agent an environment where it can call use python code to search documents using both traditional search + code-native “logic” (think regex, if/else, for loops, etc) to find answers/content. For evaluations, I will use LLM-as-a-judge in Prime-RL environments (https://app.primeintellect.ai/dashboard/environments/will/wiki-search), each response will get a binary good/bad response score, that will be judged by a separate language model viewing a curated “good” answer, and what the being-evaluated SLM outputs as the “predicted” answer.