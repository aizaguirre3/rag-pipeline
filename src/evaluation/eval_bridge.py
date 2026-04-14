from __future__ import annotations

"""Bridge between the RAG pipeline and the LLM Eval Harness.

Exports RAG pipeline results as a dataset that can be scored by the eval harness.
"""

import json
from pathlib import Path
from typing import List

from src.generation.generator import RAGResponse


def export_for_eval_harness(
    responses: List[RAGResponse],
    output_path: str,
) -> str:
    """Export RAG responses as a Q&A dataset compatible with the eval harness.

    The exported JSON can be loaded by the eval harness's DatasetLoader, and the
    results can be scored using RAGAS or custom metrics.
    """
    qa_pairs = []
    for i, r in enumerate(responses):
        qa_pairs.append({
            "id": f"rag_{i+1:03d}",
            "question": r.question,
            "expected_answer": r.answer,
            "context": r.context,
            "metadata": {
                "category": "rag_generated",
                "model": r.model,
                "sources": r.sources,
                "latency_ms": round(r.latency_ms, 1),
            },
        })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qa_pairs, indent=2))
    print(f"Exported {len(qa_pairs)} Q&A pairs to {output_path}")
    return str(path)


def export_eval_report(
    responses: List[RAGResponse],
    output_path: str,
) -> str:
    """Export a full evaluation report with retrieval and generation details."""
    report = {
        "metadata": {
            "pipeline": "rag-pipeline",
            "total_questions": len(responses),
            "model": responses[0].model if responses else "N/A",
        },
        "results": [],
        "summary": {},
    }

    total_latency = 0.0
    total_input = 0
    total_output = 0

    for r in responses:
        entry = {
            "question": r.question,
            "answer": r.answer,
            "sources": r.sources,
            "num_chunks_retrieved": len(r.retrieval_results),
            "top_chunk_score": r.retrieval_results[0].score if r.retrieval_results else 0,
            "latency_ms": round(r.latency_ms, 1),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        }
        report["results"].append(entry)
        total_latency += r.latency_ms
        total_input += r.input_tokens
        total_output += r.output_tokens

    if responses:
        report["summary"] = {
            "avg_latency_ms": round(total_latency / len(responses), 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_chunks_retrieved": round(
                sum(len(r.retrieval_results) for r in responses) / len(responses), 1
            ),
        }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"Report exported to {output_path}")
    return str(path)
