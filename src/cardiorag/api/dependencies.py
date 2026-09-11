"""Lazily-loaded, process-wide singletons for the API.

The embedding model, FAISS index, and reranker are expensive to load
(seconds, real memory) and must not be reconstructed per request. Each
getter is cached with `lru_cache`, and used as a FastAPI dependency
(`Depends(...)`) rather than a bare module-level global so tests can swap
in fakes via `app.dependency_overrides` without needing the real models.

The LLM provider is intentionally NOT loaded at import time: if it fails
(e.g. no API key configured yet), `/health`, `/retrieve`, and `/documents`
must keep working - only `/query` actually needs it.
"""

from functools import lru_cache

from cardiorag.config import settings
from cardiorag.embeddings.embedder import Embedder
from cardiorag.generation.providers import LLMProvider, RetryingProvider, load_provider_from_settings
from cardiorag.retrieval.reranker import Reranker
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore.load(settings.index_dir)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embedding_model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(get_embedder(), get_vector_store())


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Raises (ValueError/NotImplementedError) if misconfigured. Not cached
    on failure - functools.lru_cache never caches a raised exception, so a
    later request can succeed once the operator fixes the configuration
    without restarting the process. Retries here use a much smaller budget
    than the batch-evaluation scripts: an HTTP client is waiting, so a
    request should fail fast on a persistent problem rather than retry for
    up to a minute like the evaluation loop does.
    """
    return RetryingProvider(load_provider_from_settings(settings), max_retries=2, base_delay_seconds=1.0)
