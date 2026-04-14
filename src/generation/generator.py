from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import anthropic

from src.config import settings
from src.retrieval.retriever import Retriever, RetrievalResult

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer ONLY based on the provided context. Do not use prior knowledge.
- If the context doesn't contain enough information, say "I don't have enough information to answer that."
- Cite which source(s) you used when possible.
- Be concise and accurate."""


@dataclass
class RAGResponse:
    question: str
    answer: str
    context: str
    sources: List[str]
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    retrieval_results: List[RetrievalResult]


class RAGGenerator:
    """Generates answers using retrieved context and Claude."""

    def __init__(
        self,
        model: str = settings.default_model,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.retriever = Retriever()

    def generate(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Full RAG pipeline: retrieve context then generate answer."""
        # 1. Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(question, top_k)
        context = self._format_context(retrieval_results)
        sources = list({r.metadata.get("source", "unknown") for r in retrieval_results})

        # 2. Generate answer with context
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            question=question,
            answer=response.content[0].text,
            context=context,
            sources=sources,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            retrieval_results=retrieval_results,
        )

    def generate_with_context(self, question: str, context: str) -> RAGResponse:
        """Generate answer with pre-provided context (skip retrieval)."""
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            question=question,
            answer=response.content[0].text,
            context=context,
            sources=[],
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            retrieval_results=[],
        )

    def _format_context(self, results: List[RetrievalResult]) -> str:
        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source} | Relevance: {r.score:.2f}]\n{r.chunk_text}")

        return "\n\n".join(parts)
