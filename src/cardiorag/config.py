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
    # Runs entirely offline via `transformers` once cached - no API key, no
    # per-request network call, at a real quality cost vs. a much larger
    # hosted model (see README). Qwen2.5-0.5B-Instruct: small enough for
    # CPU-only inference to finish in a reasonable time for one request,
    # Apache-2.0 licensed, ungated (no HF auth token needed to download it).
    huggingface_local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    huggingface_local_device: Literal["auto", "cpu", "cuda"] = "auto"
    huggingface_local_max_new_tokens: int = Field(default=512, gt=0)

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    # LRU cache size for repeated query embeddings in the API process (0
    # disables caching entirely). Never applies to chunk/ingestion
    # embedding, which is always a batch call and passes through untouched.
    query_embedding_cache_size: int = Field(default=256, ge=0)

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

    # --- API security (Phase 20 fix #6) ---
    # A single shared secret, not per-user accounts - appropriate for a small
    # research/demo deployment with a handful of trusted clients (the
    # Streamlit UI, evaluation scripts), not a multi-tenant product. Left
    # unset by default so the pre-fix behavior (fully open access) remains
    # the default for local/dev use; an operator opts in by setting API_KEY.
    api_key: str | None = None
    # Fixed-window requests/minute per client IP, enforced on /retrieve,
    # /query and /documents. Not applied to /health, which the Docker
    # Compose healthcheck polls every 10s with plain curl and no credentials.
    rate_limit_per_minute: int = 60

    # --- Logging ---
    log_level: str = "INFO"

    def masked(self) -> dict:
        """Settings as a dict safe to log or print - never call `dict(settings)`
        or `print(settings)` directly (Phase 17: never log secrets), since
        api_key fields would appear in full. Presence is still shown (True/
        False), since knowing *whether* a key is configured is useful for
        debugging without ever exposing the key itself.
        """
        data = self.model_dump()
        for key in ("openai_api_key", "groq_api_key"):
            data[key] = bool(data[key])
        return data


settings = Settings()
