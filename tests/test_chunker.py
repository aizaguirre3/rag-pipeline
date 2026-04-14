from src.ingestion.chunker import Chunker, Chunk


def test_chunk_text_basic():
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five, which is a bit longer to push us over the chunk boundary."
    chunks = chunker.chunk_text(text, source="test.txt")

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


def test_chunk_preserves_metadata():
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text("Hello world.", source="doc.txt", metadata={"topic": "greeting"})

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "doc.txt"
    assert chunks[0].metadata["topic"] == "greeting"


def test_chunk_generates_ids():
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text("Hello world.", source="test.txt")

    assert len(chunks) == 1
    assert len(chunks[0].chunk_id) == 12


def test_empty_text():
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text("", source="empty.txt")
    assert len(chunks) == 0


def test_chunk_overlap():
    chunker = Chunker(chunk_size=50, chunk_overlap=20)
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = chunker.chunk_text(text, source="test.txt")

    # With overlap, later chunks should share some text with previous chunks
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) > 0
