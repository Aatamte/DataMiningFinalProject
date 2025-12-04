"""Core environment loading for wiki-search."""

from datasets import load_dataset

# Fixed 80/20 split by index (deterministic)
TRAIN_SPLIT_RATIO = 0.8


def load_questions(
    dataset_name: str = "willcb/wiki-trivia-questions-v4",
    subset: str = "train",
):
    """Load questions dataset with train/test/all split support.

    Uses fixed 80/20 split by index for reproducibility:
    - train: first 80% of questions (indices 0 to split_idx-1)
    - test: last 20% of questions (indices split_idx to end)
    - all: entire dataset

    Args:
        dataset_name: HuggingFace dataset ID for questions
        subset: Which subset to load - "train", "test", or "all"

    Returns:
        HuggingFace dataset with questions
    """
    # Load full dataset
    full_dataset = load_dataset(dataset_name, split="train")

    total_size = len(full_dataset)
    split_idx = int(total_size * TRAIN_SPLIT_RATIO)

    if subset == "train":
        return full_dataset.select(range(0, split_idx))
    elif subset == "test":
        return full_dataset.select(range(split_idx, total_size))
    elif subset == "all":
        return full_dataset
    else:
        raise ValueError(f"Unknown subset: {subset}. Use 'train', 'test', or 'all'")
