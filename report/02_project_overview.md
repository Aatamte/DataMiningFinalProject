## Project Overview

# Teaching Small Language Models to Search with Python

Large language models excel at recalling facts from their training data, but struggle with questions requiring information they haven't memorized. Retrieval-augmented generation (RAG) addresses this by fetching relevant documents, yet typically relies on a single query and cannot adapt when initial results prove insufficient. Recent tool-use frameworks give models more flexibility, but constrain them to rigid interfaces—one tool call at a time, with fixed input/output formats.

This work takes the next step: instead of calling tools, the model writes Python code that executes in a sandbox. Search functions are primitives the model can compose with loops, conditionals, and string operations—transforming the problem from "which tool to call" into "what program to write." The model learns entirely through reinforcement learning, discovering effective search strategies through trial and error with no human demonstrations.

Our trained agent (Qwen3-4B, 4 billion parameters) achieves TODO% accuracy on held-out questions, compared to TODO% for the untrained baseline—a TODO percentage point improvement from reinforcement learning alone. The agent learns non-trivial behaviors: iterating through search results with programmatic filters, navigating from broad topic pages to specific sections, and adapting its strategy based on intermediate outputs. These results demonstrate that small language models can learn to program effective information-seeking behavior without human demonstrations.
