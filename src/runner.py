from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.config import DOCUMENTS_DIR, PROJECT_ROOT
from src.generation.generator import RAGGenerator, RAGResponse
from src.ingestion.ingestor import DocumentIngestor
from src.retrieval.retriever import Retriever

RESULTS_DIR = PROJECT_ROOT / "results"


def ingest(directory: str = "", clear: bool = False) -> None:
    """Ingest documents into the vector store."""
    ingestor = DocumentIngestor()

    if clear:
        ingestor.clear()
        print("Cleared existing documents from vector store.")

    dir_path = directory or str(DOCUMENTS_DIR)
    print(f"Ingesting documents from: {dir_path}")
    chunks = ingestor.ingest_directory(dir_path)
    stats = ingestor.get_stats()
    print(f"\nIngestion complete. {stats['total_chunks']} total chunks in collection.")


def query(question: str, model: str = "", top_k: int = 5) -> RAGResponse:
    """Ask a question using the RAG pipeline."""
    kwargs = {}
    if model:
        kwargs["model"] = model
    generator = RAGGenerator(**kwargs)
    response = generator.generate(question, top_k=top_k)

    print(f"\nQuestion: {response.question}")
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Sources: {', '.join(response.sources)}")
    print(f"\nAnswer:\n{response.answer}")

    return response


def batch_query(
    questions_file: str,
    model: str = "",
    top_k: int = 5,
    output: str = "",
) -> List[RAGResponse]:
    """Run multiple questions through the RAG pipeline."""
    path = Path(questions_file)
    questions = json.loads(path.read_text())

    kwargs = {}
    if model:
        kwargs["model"] = model
    generator = RAGGenerator(**kwargs)

    responses: List[RAGResponse] = []
    for i, q in enumerate(questions, 1):
        question = q if isinstance(q, str) else q.get("question", "")
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")

        response = generator.generate(question, top_k=top_k)
        responses.append(response)

        print(f"  Latency: {response.latency_ms:.0f}ms | Sources: {', '.join(response.sources)}")

    _print_summary(responses)

    if output:
        from src.evaluation.eval_bridge import export_eval_report
        export_eval_report(responses, output)

    return responses


def stats() -> None:
    """Show vector store statistics."""
    ingestor = DocumentIngestor()
    s = ingestor.get_stats()
    print(f"Collection: {s['collection']}")
    print(f"Total chunks: {s['total_chunks']}")


def _print_summary(responses: List[RAGResponse]) -> None:
    print(f"\n{'='*70}")
    print("BATCH QUERY SUMMARY")
    print(f"{'='*70}")
    print(f"  Total questions: {len(responses)}")

    latencies = [r.latency_ms for r in responses]
    print(f"  Avg latency: {sum(latencies)/len(latencies):.0f}ms")
    print(f"  Total input tokens: {sum(r.input_tokens for r in responses):,}")
    print(f"  Total output tokens: {sum(r.output_tokens for r in responses):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline - Retrieve, Augment, Generate")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("-d", "--directory", default="", help="Directory of documents")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear existing documents first")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("-m", "--model", default="", help="Claude model to use")
    query_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch queries from a JSON file")
    batch_parser.add_argument("questions_file", help="JSON file with questions")
    batch_parser.add_argument("-m", "--model", default="", help="Claude model to use")
    batch_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve")
    batch_parser.add_argument("-o", "--output", default="", help="Output JSON report path")

    # Stats command
    subparsers.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(directory=args.directory, clear=args.clear)
    elif args.command == "query":
        query(question=args.question, model=args.model, top_k=args.top_k)
    elif args.command == "batch":
        batch_query(
            questions_file=args.questions_file,
            model=args.model,
            top_k=args.top_k,
            output=args.output,
        )
    elif args.command == "stats":
        stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
