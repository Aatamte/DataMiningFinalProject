## Input Data

This work uses two publicly available datasets from HuggingFace: a Wikipedia corpus for retrieval and a question-answer dataset for training and evaluation. The corpus provides the knowledge base the model must search, while the question-answer pairs define the task and supply ground truth for reward computation.

**Search Corpus:** The corpus contains 2,590 Wikipedia pages from the [rare-wiki-pages](https://huggingface.co/datasets/willcb/rare-wiki-pages) dataset. These pages cover niche topics less likely to be memorized during language model pretraining, ensuring the task tests search capability rather than recall. Pages are formatted as markdown with hierarchical section headers, providing structure the model can navigate programmatically.

```json
{
    "id": "caroline_polachek",
    "title": "Caroline Polachek",
    "content": "# Caroline Polachek\n\n**Caroline Polachek** (born 1985) is an American singer...\n\n## Early life\n...\n\n## Career\n\n### Chairlift (2008–2017)\n..."
}
```

**Question-Answer Pairs:** The evaluation set comprises 478 factoid question-answer pairs from the [wiki-trivia-questions-v4](https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4) dataset. Questions span diverse domains including entertainment, science, geography, and current events, with answers typically consisting of short noun phrases, named entities, or numerical values.

```json
{
    "question": "Which university did Miles Teller attend for his Bachelor of Fine Arts degree?",
    "answer": "New York University",
    "filename": "Miles Teller.md"
}
```

**Data Split:** We partition the dataset deterministically by index, allocating the first 80% (382 questions) for training and reserving the remaining 20% (96 questions) for evaluation. This fixed split ensures reproducibility and prevents evaluation on training examples.
