"""Semantic retrieval: turns a natural-language query into the top-k most
similar chunks in the vector index.

    query -> embed (same model the index was built with) -> FAISS search -> ranked chunks

This is the only place the query-side embedding step happens - vector_store.py
deliberately only knows about vectors, not text, so that boundary stays
explicit rather than implicit.
"""

from pathlib import Path

from cardiorag.embeddings.embedder import Embedder
from cardiorag.models import RetrievedChunk
from cardiorag.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        if embedder.dimension != vector_store.index.d:
            raise ValueError(
                f"Embedder produces {embedder.dimension}-dim vectors but the index was "
                f"built with {vector_store.index.d}-dim vectors - they must come from the "
                "same embedding model."
            )
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        query_vector = self._embedder.embed([query]).vectors[0]
        results = self._vector_store.search(query_vector, top_k=top_k)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in results]


def load_retriever(index_dir: Path, embedding_model: str, device: str = "auto") -> Retriever:
    """Convenience factory: load a persisted index and wire it up with the
    embedding model that must have produced it (same model_name settings
    used when the index was built - passing a different one will raise via
    the dimension check above, but not necessarily catch a same-dimension
    different-model mismatch, which is why the model name must match by
    convention with the index's persisted directory).
    """
    vector_store = VectorStore.load(index_dir)
    embedder = Embedder(embedding_model, device=device)
    return Retriever(embedder, vector_store)
