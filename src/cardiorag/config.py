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
    llm_provider: Literal["openai", "groq", "huggingface_local", "ollama"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    # Groq hosts open models behind an OpenAI-API-compatible endpoint, so it
    # reuses OpenAIProvider with a different base_url - no new provider class
    # needed, just different configuration (proof that Phase 9's abstraction
    # actually works across unrelated backends).
    #
    # Every model on the free tier shares the same ~200k-tokens/day ceiling
    # PER MODEL (not a shared org-wide pool) - confirmed by exhausting it on
    # four different models in a row during Phase 13's evaluation loop
    # (openai/gpt-oss-120b, openai/gpt-oss-20b, groq/compound-mini - which
    # turned out to itself route to gpt-oss-120b under the hood, sharing its
    # quota - then qwen/qwen3.8-27b). "qwen/qwen3.8-27b" is the default here
    # because it reliably followed this project's "return ONLY JSON" judge
    # instructions; "allam-2-7b" is a real fallback with a separate quota,
    # but often answered in free-form prose instead of the requested JSON,
    # which silently degrades the LLM-as-judge metrics in generation_metrics.py.
    groq_api_key: str | None = None
    groq_model: str = "qwen/qwen3.8-27b"
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
    # Used by the Streamlit UI (Phase 16) to reach the API server - not the
    # same value as api_host, which is what the server binds to (0.0.0.0
    # isn't a valid address to connect *to* from a client).
    api_base_url: str = "http://localhost:8000"

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()
