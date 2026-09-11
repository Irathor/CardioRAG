"""Hybrid retrieval: dense (Retriever) + BM25 lexical, combined by
Reciprocal Rank Fusion (Phase 20 fix #5).

A new, additive class rather than a modification to Retriever - existing
code and tests built against Retriever's exact behavior (Phases 6-19)
stay untouched. BM25Index is rebuilt in memory from the persisted chunk
list at load time rather than persisted separately: rebuilding is fast at
this corpus's size (hundreds of chunks), so there's no real index-drift
risk to guard against by serializing it.
"""

from pathlib import Path

from cardiorag.embeddings.embedder import Embedder
from cardiorag.models import RetrievedChunk
from cardiorag.retrieval.bm25_index import BM25Index, reciprocal_rank_fusion
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        candidate_k: int = 50,
        rrf_k: int = 10,  # see bm25_index.reciprocal_rank_fusion's docstring for why not 60
    ):
        self._retriever = Retriever(embedder, vector_store)
        self._bm25_index = bm25_index
        # How many candidates each side contributes before fusion - larger
        # than the final top_k so fusion has real material to combine, not
        # just re-ranking the same handful of candidates.
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        dense_results = [
            (r.chunk, r.score) for r in self._retriever.retrieve(query, top_k=self._candidate_k)
        ]
        bm25_results = self._bm25_index.search(query, top_k=self._candidate_k)

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=self._rrf_k)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in fused[:top_k]]


def load_hybrid_retriever(
    index_dir: Path, embedding_model: str, device: str = "auto"
) -> HybridRetriever:
    vector_store = VectorStore.load(index_dir)
    embedder = Embedder(embedding_model, device=device)
    bm25_index = BM25Index(vector_store.chunks)
    return HybridRetriever(embedder, vector_store, bm25_index)
