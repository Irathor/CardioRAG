from pathlib import Path

import numpy as np
import pytest

from cardiorag.embeddings.embedder import EmbeddingResult
from cardiorag.embeddings.store import load_embeddings, model_slug, save_embeddings
from cardiorag.models import Chunk


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc1-{i:04d}",
            document_id="doc1",
            text=f"chunk text {i}",
            token_count=10 + i,
            page_numbers=[1],
            title="A Test Paper",
            doi="10.1/test",
            source_filename="synthetic.pdf",
        )
        for i in range(n)
    ]


def test_model_slug_replaces_slashes():
    assert model_slug("sentence-transformers/all-MiniLM-L6-v2") == (
        "sentence-transformers__all-MiniLM-L6-v2"
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    chunks = _make_chunks(3)
    vectors = np.random.default_rng(0).random((3, 8)).astype(np.float32)
    result = EmbeddingResult(
        vectors=vectors,
        model_name="fake/model",
        dimension=8,
        normalized=True,
        encode_seconds=0.1,
    )

    save_embeddings(tmp_path, chunks, result)
    loaded_vectors, loaded_chunks = load_embeddings(tmp_path, "fake/model")

    np.testing.assert_allclose(loaded_vectors, vectors)
    assert [c.chunk_id for c in loaded_chunks] == [c.chunk_id for c in chunks]
    assert loaded_chunks == chunks


def test_save_raises_on_mismatched_lengths(tmp_path: Path):
    chunks = _make_chunks(3)
    result = EmbeddingResult(
        vectors=np.zeros((2, 8), dtype=np.float32),
        model_name="fake/model",
        dimension=8,
        normalized=True,
        encode_seconds=0.0,
    )

    with pytest.raises(ValueError):
        save_embeddings(tmp_path, chunks, result)


def test_load_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path, "nonexistent/model")
