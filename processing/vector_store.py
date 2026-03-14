import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import chromadb
from chromadb.config import Settings
from models import Chunk
from processing.embedder import embed_texts, embed_query



class VectorStore:
    def __init__(self, collection_name: str = "transcript") -> None:
        self._client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        texts      = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        self._collection.add(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [
                {"timestamp": c.timestamp_str, "start": c.start}
                for c in chunks
            ],
        )

    def query(self, query_text: str, top_k: int = 8) -> list[dict]:

        n = min(top_k, self._collection.count())
        if n == 0:
            return []

        results = self._collection.query(
            query_embeddings = [embed_query(query_text)],
            n_results        = n,
            include          = ["documents", "metadatas", "distances"],
        )

        chunks = [
            {
                "text":      doc,
                "timestamp": meta.get("timestamp", ""),
                "start":     meta.get("start", 0.0),
                "distance":  dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        chunks.sort(key=lambda c: c["start"])
        return chunks

    def count(self) -> int:
        return self._collection.count()