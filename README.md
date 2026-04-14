# RAG Pipeline

A production-ready Retrieval-Augmented Generation pipeline built with ChromaDB and Claude. Ingests documents, chunks them with sentence-aware boundaries, stores embeddings in a vector database, and generates grounded answers using Claude.

## Architecture

```
Documents (.txt/.md)
    ↓
┌─────────────────────────────────┐
│  Ingestion                      │
│  ├─ Document Loading            │
│  ├─ Sentence-Aware Chunking     │
│  └─ ChromaDB Vector Storage     │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Retrieval                      │
│  ├─ Semantic Search (Cosine)    │
│  ├─ Top-K Chunk Selection       │
│  └─ Context Formatting          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Generation                     │
│  ├─ Context-Augmented Prompting │
│  ├─ Claude API                  │
│  └─ Source Attribution          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Evaluation                     │
│  └─ Export to LLM Eval Harness  │
└─────────────────────────────────┘
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/aizaguirre3/rag-pipeline.git
cd rag-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set your API key
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Ingest documents
python -m src.runner ingest

# Ask a question
python -m src.runner query "What is the transformer architecture?"

# Run batch queries
python -m src.runner batch examples/questions.json -o results/batch_run.json

# Check vector store stats
python -m src.runner stats
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Ingest documents from `/data/documents/` |
| `ingest --clear` | Clear vector store and re-ingest |
| `ingest -d /path/to/docs` | Ingest from custom directory |
| `query "question"` | Ask a single question |
| `query "question" -k 3` | Retrieve top 3 chunks |
| `batch questions.json` | Run batch evaluation |
| `batch questions.json -o report.json` | Export results |
| `stats` | Show vector store statistics |

## Key Design Decisions

- **Sentence-aware chunking** with configurable size/overlap prevents splitting mid-sentence
- **ChromaDB** for local, zero-config vector storage with cosine similarity
- **Deterministic chunk IDs** (SHA-256 hash) enable idempotent upserts
- **Source attribution** in generated answers for traceability
- **Eval harness integration** — export RAG results as datasets for RAGAS scoring

## Integration with LLM Eval Harness

This project pairs with the [LLM Eval Harness](https://github.com/aizaguirre3/llm-eval-harness) to create a complete build-measure-optimize loop:

1. **Build**: This RAG pipeline generates answers from retrieved context
2. **Measure**: The eval harness scores those answers with RAGAS metrics (faithfulness, context precision, context recall)
3. **Optimize**: Regression testing catches quality drops when you change chunking, retrieval, or prompts

## Stack

- **Python 3.9+**
- **Anthropic Claude API** — LLM for generation
- **ChromaDB** — Vector database with built-in embeddings
- **Pydantic** — Data validation and settings management

## Tests

```bash
pytest tests/ -v
```

15 tests covering chunking, ingestion, retrieval, and generation.
