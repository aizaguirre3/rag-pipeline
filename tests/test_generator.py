from unittest.mock import MagicMock, patch

from src.generation.generator import RAGGenerator, RAGResponse
from src.retrieval.retriever import RetrievalResult


@patch("src.generation.generator.Retriever")
@patch("src.generation.generator.anthropic.Anthropic")
def test_generate_returns_rag_response(mock_anthropic_cls, mock_retriever_cls):
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk_text="Python is great for ML.", score=0.9, metadata={"source": "test.txt"})
    ]
    mock_retriever_cls.return_value = mock_retriever

    # Mock Anthropic client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Python is great for ML because of its libraries.")]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 20

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    generator = RAGGenerator()
    response = generator.generate("Why is Python good for ML?")

    assert isinstance(response, RAGResponse)
    assert response.question == "Why is Python good for ML?"
    assert "Python" in response.answer
    assert len(response.retrieval_results) == 1


@patch("src.generation.generator.anthropic.Anthropic")
def test_generate_with_context(mock_anthropic_cls):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The answer is 42.")]
    mock_response.usage.input_tokens = 30
    mock_response.usage.output_tokens = 10

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    generator = RAGGenerator()
    response = generator.generate_with_context("What is the answer?", "The answer to everything is 42.")

    assert response.answer == "The answer is 42."
    assert response.retrieval_results == []
