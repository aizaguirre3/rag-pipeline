import pytest

from src.ingestion.ingestor import DocumentIngestor
from src.retrieval.retriever import Retriever, RetrievalResult


@pytest.fixture
def populated_store(tmp_path):
    persist_dir = str(tmp_path / "vectorstore")
    ingestor = DocumentIngestor(
        collection_name="test_retrieval",
        persist_dir=persist_dir,
    )
    ingestor.ingest_text(
        "Python is a programming language used for data science and machine learning. "
        "It has libraries like NumPy, Pandas, and scikit-learn.",
        source="python.txt",
    )
    ingestor.ingest_text(
        "JavaScript is a programming language primarily used for web development. "
        "Popular frameworks include React, Vue, and Angular.",
        source="javascript.txt",
    )
    return persist_dir


def test_retrieve_returns_results(populated_store):
    retriever = Retriever(
        collection_name="test_retrieval",
        persist_dir=populated_store,
        top_k=2,
    )
    results = retriever.retrieve("machine learning with Python")

    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)


def test_retrieve_relevance_ordering(populated_store):
    retriever = Retriever(
        collection_name="test_retrieval",
        persist_dir=populated_store,
        top_k=2,
    )
    results = retriever.retrieve("Python data science")

    # The Python-related chunk should score higher
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_retrieve_with_context(populated_store):
    retriever = Retriever(
        collection_name="test_retrieval",
        persist_dir=populated_store,
        top_k=1,
    )
    context = retriever.retrieve_with_context("web development JavaScript")

    assert "JavaScript" in context or "javascript" in context.lower()


def test_retrieve_empty_collection(tmp_path):
    retriever = Retriever(
        collection_name="empty_collection",
        persist_dir=str(tmp_path / "empty_store"),
    )
    results = retriever.retrieve("anything")
    assert results == []
