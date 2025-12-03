"""Core environment loading for wiki-search."""

from datasets import load_dataset


def load_questions(
    dataset_name: str = "willcb/wiki-trivia-questions-v4",
    split: str = "train",
):
    """Load just the questions dataset for training.

    All tool execution goes through the Docker sandbox server.
    This function only loads the questions - no corpus, no ChromaDB.

    Args:
        dataset_name: HuggingFace dataset ID for questions
        split: Dataset split to use

    Returns:
        HuggingFace dataset with questions
    """
    return load_dataset(dataset_name, split=split)
