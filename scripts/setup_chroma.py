"""One-time setup script to index wiki pages into ChromaDB."""

import sys
sys.path.insert(0, ".")

from src.environment import load_environment


def main():
    print("Setting up ChromaDB index with local embeddings...")
    print("This will download the dataset and create embeddings.")
    print("First run may take 5-10 minutes.\n")

    # Loading the environment will trigger indexing
    env = load_environment(
        chroma_db_dir="data/.chroma_db",
        embed_model="all-MiniLM-L6-v2",
        embed_device="cuda",
    )

    print("\nSetup complete!")
    print(f"Dataset size: {len(env.dataset)} questions")


if __name__ == "__main__":
    main()
