import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

load_dotenv(PROJECT_ROOT / ".env", override=True)


class Settings(BaseSettings):
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    default_model: str = "claude-sonnet-4-20250514"
    chunk_size: int = 500
    chunk_overlap: int = 50
    collection_name: str = "rag_documents"
    top_k: int = 5

settings = Settings()
