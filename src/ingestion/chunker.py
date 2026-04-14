from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.config import settings


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(self.text.encode()).hexdigest()[:12]


class Chunker:
    """Splits documents into overlapping text chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source: str = "",
        metadata: Dict[str, str] | None = None,
    ) -> List[Chunk]:
        """Split text into chunks using sentence-aware boundaries."""
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_meta = {"source": source, "chunk_index": str(len(chunks))}
                if metadata:
                    chunk_meta.update(metadata)

                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

                # Keep overlap sentences
                overlap_text = ""
                overlap_sentences: List[str] = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_text = " ".join(overlap_sentences)

                current_chunk = overlap_sentences
                current_length = len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_len

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_meta = {"source": source, "chunk_index": str(len(chunks))}
            if metadata:
                chunk_meta.update(metadata)
            chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
