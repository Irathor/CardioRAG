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

from fastapi import Depends, HTTPException, Request

from cardiorag.api.rate_limit import RateLimiter
from cardiorag.config import settings
from cardiorag.embeddings.caching_embedder import CachingEmbedder
from cardiorag.embeddings.embedder import Embedder
from cardiorag.generation.providers import (
    LLMProvider,
    RetryingProvider,
    load_provider_from_settings,
)
from cardiorag.retrieval.bm25_index import BM25Index
from cardiorag.retrieval.hybrid_retriever import HybridRetriever
from cardiorag.retrieval.reranker import Reranker
from cardiorag.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore.load(settings.index_dir)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder | CachingEmbedder:
    """Wrapped in a query-embedding cache (project improvement round) when
    settings.query_embedding_cache_size > 0 - the API process is the one
    place a query genuinely repeats (the same or a similar question from
    different users/UI sessions hitting the same long-lived process), unlike
    a one-off script or a test run that starts with an empty cache anyway.
    """
    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    if settings.query_embedding_cache_size > 0:
        return CachingEmbedder(embedder, maxsize=settings.query_embedding_cache_size)
    return embedder


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    return BM25Index(get_vector_store().chunks)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Hybrid (dense + BM25) retrieval, not plain dense - measured on the
    real 34-question evaluation set to beat dense-only on every metric
    (Hit Rate 0.82->0.94, MRR 0.60->0.72, nDCG 0.30->0.40), so it's the
    default here rather than an opt-in alternative. See README Technical
    Decisions."""
    return HybridRetriever(get_embedder(), get_vector_store(), get_bm25_index())


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(settings.rate_limit_per_minute)


def enforce_rate_limit(request: Request, limiter: RateLimiter = Depends(get_rate_limiter)) -> None:
    client_id = request.client.host if request.client else "unknown"
    if not limiter.allow(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


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
