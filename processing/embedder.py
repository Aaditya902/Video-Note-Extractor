"""
processing/embedder.py — Embed text chunks using sentence-transformers.

Uses 'all-MiniLM-L6-v2': fast, 384-dim, great for semantic similarity.
Model is downloaded once and cached locally by HuggingFace.
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load embedding model once and cache it for the session."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts: List of strings to embed

    Returns:
        List of embedding vectors (list of floats)
    """
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used at retrieval time)."""
    return embed_texts([query])[0]