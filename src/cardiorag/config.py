"""Typed application configuration, loaded from environment variables / .env.

Every other module should import `settings` from here rather than calling
`os.getenv` directly. This gives us one validated, typed place to see what
the system depends on, and makes it trivial to override values in tests.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["openai", "huggingface_local", "ollama"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Reranking ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, gt=0)

    # --- Paths ---
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("data/corpus")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("indexes")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()
