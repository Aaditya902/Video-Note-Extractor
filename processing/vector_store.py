"""
processing/vector_store.py — In-memory ChromaDB vector store.

Each run gets its own ephemeral collection. We don't persist between runs
by default (no disk I/O needed for a CLI tool), but persistence can be
enabled by swapping to a PersistentClient.
"""

import chromadb
from chromadb.config import Settings

import sys
from pathlib import Path as _root_path
sys.path.insert(0, str(_root_path(__file__).resolve().parent.parent))

from models import Chunk
from processing.embedder import embed_texts, embed_query


class VectorStore:
    def __init__(self, collection_name: str = "transcript"):
        # In-memory client — no files written
        self._client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        # Fresh collection every run
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store all chunks in the collection."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"timestamp": c.timestamp_str, "start": c.start}
                for c in chunks
            ],
        )

    def query(self, query_text: str, top_k: int = 8) -> list[dict]:
        """
        Retrieve the top_k most semantically similar chunks.

        Returns:
            List of dicts with keys: text, timestamp, start, distance
        """
        query_embedding = embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": doc,
                "timestamp": meta.get("timestamp", ""),
                "start": meta.get("start", 0.0),
                "distance": dist,
            })

        # Sort by position in video (not by relevance score) for narrative order
        output.sort(key=lambda x: x["start"])
        return output

    def count(self) -> int:
        return self._collection.count()