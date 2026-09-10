"""Second-stage reranking with a cross-encoder.

Dense retrieval (the bi-encoder in embedder.py) embeds the query and each
chunk independently, then compares vectors - fast and precomputable into
the FAISS index, but the model never sees query and passage together.

A cross-encoder concatenates [query, passage] and scores them jointly
through the transformer's attention. It is much better at judging genuine
relevance ("does this text actually answer the question") rather than
surface vocabulary overlap - but it cannot be precomputed: it must run once
per (query, candidate) pair, so it only makes sense AFTER dense retrieval
has already narrowed the field down from the whole corpus to a small,
manageable set of candidates.

Pipeline: query -> dense retrieval (top-N) -> cross-encoder rerank -> top-K
"""

import logging
import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from cardiorag.embeddings.embedder import resolve_device
from cardiorag.models import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    reranked: list[RetrievedChunk]  # .score is now the cross-encoder's relevance
    # score, NOT a cosine similarity - a different scale than pre-rerank scores.
    rerank_seconds: float


class Reranker:
    def __init__(self, model_name: str, device: str = "auto"):
        resolved_device = resolve_device(device)
        logger.info("Loading cross-encoder %r on device=%s", model_name, resolved_device)
        self._model = CrossEncoder(model_name, device=resolved_device)
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 5
    ) -> RerankResult:
        """Score every candidate jointly with the query and return the
        top_k, re-ranked by that score. `candidates` are typically the
        top-N from dense retrieval (e.g. 20), and top_k the final context
        size (e.g. 5) - reranking is for narrowing an already-small set,
        not searching the whole corpus.
        """
        if not candidates:
            return RerankResult(reranked=[], rerank_seconds=0.0)

        start = time.perf_counter()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        elapsed = time.perf_counter() - start

        reranked = sorted(
            (
                RetrievedChunk(chunk=c.chunk, score=float(s))
                for c, s in zip(candidates, scores)
            ),
            key=lambda r: r.score,
            reverse=True,
        )[:top_k]

        logger.info("Reranked %d candidates in %.3fs", len(candidates), elapsed)
        return RerankResult(reranked=reranked, rerank_seconds=elapsed)


def retrieve_and_rerank(
    retriever, reranker: Reranker, query: str, retrieve_k: int = 20, final_k: int = 5
) -> RerankResult:
    """Convenience wiring for the full two-stage pipeline: cheap dense
    retrieval over a wide candidate pool, then expensive cross-encoder
    reranking narrows it to the final context."""
    candidates = retriever.retrieve(query, top_k=retrieve_k)
    return reranker.rerank(query, candidates, top_k=final_k)
