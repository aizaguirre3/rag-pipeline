import pytest

from src.ingestion.ingestor import DocumentIngestor


@pytest.fixture
def ingestor(tmp_path):
    return DocumentIngestor(
        collection_name="test_collection",
        persist_dir=str(tmp_path / "vectorstore"),
    )


def test_ingest_text(ingestor):
    chunks = ingestor.ingest_text("This is a test document about AI and machine learning.")
    assert len(chunks) >= 1
    stats = ingestor.get_stats()
    assert stats["total_chunks"] == len(chunks)


def test_ingest_file(ingestor, tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning.")

    chunks = ingestor.ingest_file(str(doc))
    assert len(chunks) >= 1


def test_ingest_missing_file(ingestor):
    with pytest.raises(FileNotFoundError):
        ingestor.ingest_file("/nonexistent/path.txt")


def test_clear(ingestor):
    ingestor.ingest_text("Some test content.")
    assert ingestor.get_stats()["total_chunks"] > 0

    ingestor.clear()
    assert ingestor.get_stats()["total_chunks"] == 0
