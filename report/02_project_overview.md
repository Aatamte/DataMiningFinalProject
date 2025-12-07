## Project Overview

# Teaching Small Language Models to Search with Python

Large language models excel at recalling facts from their training data, but struggle with questions requiring information they haven't memorized. Retrieval-augmented generation addresses this by fetching relevant documents, yet typically relies on a single query and cannot adapt when initial results prove insufficient. This work takes the next step: training a small language model to write and execute Python code against a search API, observe results, and iteratively refine its strategy—learning to become an effective information-seeking agent.

We train a 4-billion parameter model (Qwen3-4B) to generate Python code that searches a Wikipedia corpus through semantic search and hierarchical navigation. Given a question, the model writes code to find relevant articles, examine their structure, and extract specific content. The model learns entirely through reinforcement learning with no human demonstrations—it discovers effective search strategies through trial and error alone.

Our trained agent achieves TODO% accuracy on held-out questions, compared to TODO% for the untrained baseline—a TODO percentage point improvement from reinforcement learning alone. The agent learns non-trivial behaviors: reformulating failed queries, navigating from broad topic pages to specific sections, and composing multiple search operations within a single code block. These results demonstrate that small language models can learn effective information-seeking behavior without requiring human demonstrations of search strategies.
