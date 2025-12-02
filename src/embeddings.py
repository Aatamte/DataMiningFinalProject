"""Local embedding functions for ChromaDB."""

from chromadb.utils import embedding_functions


def get_local_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
):
    """Get a local sentence-transformers embedding function for ChromaDB.

    Args:
        model_name: Name of the sentence-transformers model to use.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        A ChromaDB-compatible embedding function.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )
