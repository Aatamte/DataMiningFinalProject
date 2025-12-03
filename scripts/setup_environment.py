"""Setup the execution environment.

This script:
1. Downloads the Wikipedia corpus from HuggingFace
2. Saves it to data/corpus/corpus.json
3. Builds the ChromaDB index in data/.chroma_db/
4. Builds the Docker image for the execution environment

Run once before starting the execution environment:
    uv run python scripts/setup_environment.py

Options:
    --data-dir PATH   Use existing data directory (skips download/indexing if data exists)
    --skip-index      Skip ChromaDB indexing entirely
    --docker-only     Only build Docker image, skip all data setup

Environment variables:
    DEV=1             Development mode - only index 10 pages for faster testing
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()


# Configuration
CORPUS_DATASET = "willcb/rare-wiki-pages"
CORPUS_SPLIT = "train"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Dev mode - limit corpus size for faster testing
DEV_MODE = os.environ.get("DEV", "").lower() in ("1", "true", "yes")
DEV_CORPUS_LIMIT = 10


def setup_directories(data_dir: Path):
    """Create necessary directories."""
    print("Creating directories...")
    (data_dir / "corpus").mkdir(parents=True, exist_ok=True)
    (data_dir / ".chroma_db").mkdir(parents=True, exist_ok=True)
    print("  Done.")


def download_corpus(data_dir: Path):
    """Download corpus from HuggingFace and save as JSON."""
    corpus_path = data_dir / "corpus" / "corpus.json"
    print(f"\nDownloading corpus from {CORPUS_DATASET}...")

    if corpus_path.exists():
        print(f"  Corpus already exists at {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"  Loaded {len(corpus)} pages from cache.")
    else:
        # Lazy import - only needed if downloading
        from datasets import load_dataset
        dataset = load_dataset(CORPUS_DATASET, split=CORPUS_SPLIT)

        corpus = []
        for row in dataset:
            corpus.append({
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
            })

        print(f"  Downloaded {len(corpus)} pages.")

        print(f"  Saving to {corpus_path}...")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)

        print("  Done.")

    # Limit corpus in dev mode
    if DEV_MODE:
        corpus = corpus[:DEV_CORPUS_LIMIT]
        print(f"  [DEV MODE] Limited to {len(corpus)} pages.")

    return corpus


def build_chroma_index(corpus: list[dict], data_dir: Path):
    """Build ChromaDB index for the corpus."""
    chroma_dir = str(data_dir / ".chroma_db")
    print(f"\nBuilding ChromaDB index in {chroma_dir}...")

    # Lazy imports - only needed if indexing
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    print("  Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    class LocalEmbeddingFunction:
        def name(self) -> str:
            return "local-minilm"

        def __call__(self, input: list[str]) -> list[list[float]]:
            embeddings = embed_model.encode(input)
            return embeddings.tolist()

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete existing collection to avoid metadata issues, then recreate
    try:
        client.delete_collection(name="wiki_titles")
        print("  Deleted existing collection.")
    except ValueError:
        pass  # Collection doesn't exist

    # Create fresh collection WITHOUT embedding function
    collection = client.create_collection(name="wiki_titles")
    print(f"  Created fresh collection. Indexing {len(corpus)} pages...")

    # Index all pages
    missing = corpus

    # Index in batches
    batch_size = 100
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]

        ids = [page["id"] for page in batch]
        documents = [page["title"] for page in batch]
        metadatas = [{"title": page["title"]} for page in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"    Indexed batch {i // batch_size + 1}/{(len(missing) + batch_size - 1) // batch_size}")

    print("  Done.")


def build_docker_image():
    """Build the Docker image for the execution environment."""
    print(f"\nBuilding Docker image via docker-compose...")

    compose_file = Path(__file__).parent.parent / "docker" / "docker-compose.yml"

    # Use docker-compose build to ensure consistency with how we run the container
    result = subprocess.run(
        ["docker-compose", "-f", str(compose_file), "build", "--no-cache"],
    )

    if result.returncode != 0:
        print("  ERROR: Docker build failed")
        sys.exit(1)

    print("  Done.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup the execution environment for SLM-RL Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup (download, index, build docker)
  uv run python scripts/setup_environment.py

  # Use existing data directory
  uv run python scripts/setup_environment.py --data-dir /path/to/data

  # Skip indexing (data already indexed)
  uv run python scripts/setup_environment.py --skip-index

  # Only rebuild docker image
  uv run python scripts/setup_environment.py --docker-only
""",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: data/)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip ChromaDB indexing (assumes index already exists)",
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help="Only build Docker image, skip all data setup",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()

    print("=" * 60)
    print("EXECUTION ENVIRONMENT SETUP")
    if DEV_MODE:
        print(f"[DEV MODE] Limited to {DEV_CORPUS_LIMIT} pages")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    if args.docker_only:
        print("\n[--docker-only] Skipping data setup...")
        build_docker_image()
    else:
        setup_directories(data_dir)
        corpus = download_corpus(data_dir)

        if args.skip_index:
            print("\n[--skip-index] Skipping ChromaDB indexing...")
        else:
            build_chroma_index(corpus, data_dir)

        build_docker_image()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo start the execution environment:")
    print("  uv run python scripts/run_environment.py")
    print("\nTo check status:")
    print("  curl http://localhost:8080/health")


if __name__ == "__main__":
    main()
