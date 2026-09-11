import numpy as np
import pytest

from cardiorag.models import Chunk
from cardiorag.retrieval.vector_store import VectorStore, build_vector_store


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"text for {chunk_id}",
        token_count=5,
        page_numbers=[1],
        title="A Test Paper",
        doi=None,
        source_filename="synthetic.pdf",
    )


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_build_vector_store_creates_index_with_correct_size():
    vectors = _normalized([[1, 0], [0, 1], [-1, 0]])
    chunks = [_make_chunk(f"c{i}") for i in range(3)]

    store = build_vector_store(vectors, chunks)

    assert store.index.ntotal == 3
    assert store.chunks == chunks


def test_build_vector_store_rejects_length_mismatch():
    vectors = _normalized([[1, 0], [0, 1]])
    chunks = [_make_chunk("c0")]

    with pytest.raises(ValueError, match="length mismatch"):
        build_vector_store(vectors, chunks)


def test_build_vector_store_rejects_non_normalized_vectors():
    vectors = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5, not 1
    chunks = [_make_chunk("c0")]

    with pytest.raises(ValueError, match="not L2-normalized"):
        build_vector_store(vectors, chunks)


def test_build_vector_store_handles_empty_input():
    store = build_vector_store(np.empty((0, 4), dtype=np.float32), [])

    assert store.index.ntotal == 0
    assert store.search(np.zeros(4), top_k=5) == []


def test_vector_store_post_init_rejects_mismatched_index_and_chunks():
    store = build_vector_store(_normalized([[1, 0]]), [_make_chunk("c0")])

    with pytest.raises(ValueError):
        VectorStore(index=store.index, chunks=[_make_chunk("c0"), _make_chunk("c1")])


def test_search_returns_results_ordered_by_similarity_descending():
    vectors = _normalized([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]])
    chunks = [_make_chunk(f"c{i}") for i in range(4)]
    store = build_vector_store(vectors, chunks)

    query = _normalized([[1, 0]])[0]
    results = store.search(query, top_k=4)

    assert [c.chunk_id for c, _ in results] == ["c0", "c1", "c2", "c3"]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    # The exact match should score ~1.0 (cosine similarity of a vector with itself)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_search_clamps_top_k_to_available_vectors():
    vectors = _normalized([[1, 0], [0, 1]])
    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    store = build_vector_store(vectors, chunks)

    results = store.search(_normalized([[1, 0]])[0], top_k=10)

    assert len(results) == 2


def test_save_and_load_roundtrip(tmp_path):
    vectors = _normalized([[1, 0], [0, 1]])
    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    store = build_vector_store(vectors, chunks)

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)

    assert loaded.index.ntotal == 2
    assert loaded.chunks == chunks
    results = loaded.search(_normalized([[1, 0]])[0], top_k=1)
    assert results[0][0].chunk_id == "c0"


def test_load_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path)
