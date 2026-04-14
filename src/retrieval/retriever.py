from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import chromadb

from src.config import VECTORSTORE_DIR, settings


@dataclass
class RetrievalResult:
    chunk_text: str
    score: float
    metadata: Dict[str, str]


class Retriever:
    """Retrieves relevant chunks from ChromaDB for a given query."""

    def __init__(
        self,
        collection_name: str = settings.collection_name,
        persist_dir: str = str(VECTORSTORE_DIR),
        top_k: int = settings.top_k,
    ):
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        k = top_k or self.top_k
        count = self.collection.count()
        if count == 0:
            return []

        # Don't request more than available
        k = min(k, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: List[RetrievalResult] = []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            retrieval_results.append(
                RetrievalResult(chunk_text=doc, score=similarity, metadata=meta)
            )

        return retrieval_results

    def retrieve_with_context(self, query: str, top_k: int | None = None) -> str:
        """Retrieve chunks and format them as a single context string."""
        results = self.retrieve(query, top_k)
        if not results:
            return ""

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{r.chunk_text}")

        return "\n\n".join(context_parts)
