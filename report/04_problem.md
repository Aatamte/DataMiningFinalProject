## Problem

Given a factoid question and access to a Wikipedia corpus, the task is to generate a correct answer by executing search operations over the corpus. The model receives no direct access to article content; it must issue queries, interpret results, and navigate to relevant sections through code execution.

We frame this as a reinforcement learning problem:

- **State:** The conversation history—the original question, any code the model has generated, and the outputs returned by the execution environment.

- **Actions:** Text generation in two formats: Python code wrapped in `<python>...</python>` tags for execution, or a final answer wrapped in `<answer>...</answer>` tags to terminate the episode.

- **Environment:** The model interacts with the corpus through three functions:
  - `search_pages(query)` — returns articles whose titles are semantically similar to the query
  - `view_sections(page_id)` — lists the hierarchical structure of an article
  - `read_section(section_id)` — retrieves the content of a specific section

Each episode runs for up to 2 turns. The model must learn to compose these primitives into effective multi-step search strategies that locate and extract the correct answer.
