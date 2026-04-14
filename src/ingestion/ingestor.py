from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb

from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR, settings
from src.ingestion.chunker import Chunk, Chunker


class DocumentIngestor:
    """Ingests documents: reads files, chunks text, and stores embeddings in ChromaDB."""

    def __init__(
        self,
        collection_name: str = settings.collection_name,
        persist_dir: str = str(VECTORSTORE_DIR),
    ):
        self.chunker = Chunker()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_file(self, file_path: str) -> List[Chunk]:
        """Read a text file, chunk it, and add to the vector store."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = path.read_text(encoding="utf-8")
        chunks = self.chunker.chunk_text(text, source=path.name)

        self._add_chunks(chunks)
        return chunks

    def ingest_directory(self, dir_path: str = str(DOCUMENTS_DIR)) -> List[Chunk]:
        """Ingest all .txt and .md files from a directory."""
        path = Path(dir_path)
        all_chunks: List[Chunk] = []

        for ext in ("*.txt", "*.md"):
            for file_path in sorted(path.glob(ext)):
                chunks = self.ingest_file(str(file_path))
                all_chunks.extend(chunks)
                print(f"  Ingested {file_path.name}: {len(chunks)} chunks")

        print(f"Total: {len(all_chunks)} chunks from {path}")
        return all_chunks

    def ingest_text(self, text: str, source: str = "direct_input") -> List[Chunk]:
        """Chunk and ingest raw text directly."""
        chunks = self.chunker.chunk_text(text, source=source)
        self._add_chunks(chunks)
        return chunks

    def _add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to ChromaDB collection."""
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "collection": self.collection.name,
            "total_chunks": self.collection.count(),
        }

    def clear(self) -> None:
        """Delete all documents from the collection."""
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
