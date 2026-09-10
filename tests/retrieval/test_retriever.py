from types import SimpleNamespace

import numpy as np
import pytest

from cardiorag.models import Chunk
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import build_vector_store


class _FakeEmbedder:
    """Deterministic stand-in for Embedder: maps known query strings to
    fixed vectors, so Retriever's orchestration can be tested without
    loading a real model."""

    def __init__(self, vector_for_text: dict[str, list[float]], dimension: int):
        self.dimension = dimension
        self._vector_for_text = vector_for_text

    def embed(self, texts: list[str]):
        vectors = np.array([self._vector_for_text[t] for t in texts], dtype=np.float32)
        return SimpleNamespace(vectors=vectors)


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"text for {chunk_id}",
        token_count=5,
        page_numbers=[1],
        title="A Test Paper",
        doi="10.1/test",
        source_filename="synthetic.pdf",
    )


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_retrieve_returns_chunks_ranked_by_similarity():
    vectors = _normalized([[1, 0], [0, 1], [-1, 0]])
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    store = build_vector_store(vectors, chunks)
    embedder = _FakeEmbedder({"heart imaging": [1, 0]}, dimension=2)
    retriever = Retriever(embedder, store)

    results = retriever.retrieve("heart imaging", top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["c0", "c1"]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert all(isinstance(r.score, float) for r in results)


def test_retrieve_rejects_empty_query():
    store = build_vector_store(_normalized([[1, 0]]), [_make_chunk("c0")])
    retriever = Retriever(_FakeEmbedder({}, dimension=2), store)

    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ")


def test_retriever_rejects_dimension_mismatch_between_embedder_and_index():
    store = build_vector_store(_normalized([[1, 0]]), [_make_chunk("c0")])
    mismatched_embedder = _FakeEmbedder({}, dimension=3)

    with pytest.raises(ValueError, match="dim"):
        Retriever(mismatched_embedder, store)


def test_retrieve_passes_top_k_through_to_the_index():
    vectors = _normalized([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]])
    chunks = [_make_chunk(f"c{i}") for i in range(4)]
    store = build_vector_store(vectors, chunks)
    embedder = _FakeEmbedder({"q": [1, 0]}, dimension=2)
    retriever = Retriever(embedder, store)

    assert len(retriever.retrieve("q", top_k=1)) == 1
    assert len(retriever.retrieve("q", top_k=3)) == 3


# --- Integration test against the real embedding model ---


def test_retrieve_with_real_embedder_surfaces_the_relevant_chunk(embedder):
    cardiac_chunk = _make_chunk("cardiac")
    unrelated_chunk = _make_chunk("finance")
    texts = [
        "Cardiac magnetic resonance imaging assesses myocardial function and scarring.",
        "Quarterly stock market earnings reports rose sharply this year.",
    ]
    vectors = embedder.embed(texts).vectors
    store = build_vector_store(vectors, [cardiac_chunk, unrelated_chunk])
    retriever = Retriever(embedder, store)

    results = retriever.retrieve("How is heart tissue damage evaluated with MRI?", top_k=2)

    assert results[0].chunk.chunk_id == "cardiac"
    assert results[0].score > results[1].score
