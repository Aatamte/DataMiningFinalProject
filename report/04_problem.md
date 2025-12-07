## Problem

Given a factoid question and a Wikipedia corpus, generate the correct answer by writing and executing Python code.

The model has no direct access to article content. It must write code that calls search primitives—`search_pages(query)`, `view_sections(page_id)`, `read_section(section_id)`—to locate and extract the answer. Each episode allows up to 2 turns of code execution before requiring a final answer.

Traditional tool-use frameworks constrain models to one action per turn with fixed input/output schemas. Code generation enables composition—the model can loop through results, filter with conditionals, and chain operations in a single generation. This shifts the problem from "which tool to call" to "what program to write." Additionally, unlike large models that can rely on memorized facts, a 4B parameter model lacks sufficient capacity to store the corpus. It must genuinely learn to search rather than recall, making this a test of learned information-seeking behavior rather than knowledge retrieval.
