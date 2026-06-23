"""Application settings loaded from environment / .env file with extra field ignoring."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env before anything else reads it.
# Third-party clients (Anthropic, OpenAI, LangSmith) read env vars directly,
# not from the Settings object, so this must happen at import time.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"

    # LLM
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-5-20250929"
    vision_model_name: str = "claude-haiku-4-5-20251001"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 3

    # Memory — how many recent Q/A pairs to keep in a session's history
    memory_max_pairs: int = 5

    # Paths — root of the on-disk data layout; per-collection/per-PDF paths are
    # composed from this in src/registry.py.
    data_dir: str = "data"

    # LangSmith (auto-traced when LANGSMITH_TRACING=true is in .env)
    langsmith_tracing: str = ""
    langsmith_endpoint: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = ""


settings = Settings()
