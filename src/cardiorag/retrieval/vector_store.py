"""Persistent FAISS vector index over chunk embeddings, kept in lockstep
with the chunk metadata each vector belongs to.

Similarity metric: every embedding this project produces is L2-normalized
(see embeddings/embedder.py), so cosine similarity between two vectors is
exactly their inner product: cos(a, b) = a . b / (|a| |b|), and |a| = |b| = 1
here, so cos(a, b) = a . b. We therefore use FAISS's IndexFlatIP (exact
search, inner product) rather than a cosine-specific index - it computes
cosine similarity directly for normalized vectors, with no extra work.

IndexFlatIP does exact (brute-force) search. Approximate indexes (IVF,
HNSW) trade a little accuracy for speed at large scale, but only pay off
with corpora far bigger than this one - at a few hundred to a few thousand
chunks, exact search is already fast and simpler to reason about.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from cardiorag.models import Chunk

# How far a vector's L2 norm may drift from 1.0 and still count as normalized.
_NORMALIZATION_TOLERANCE = 1e-3


@dataclass
class VectorStore:
    """A FAISS index plus the chunk list it was built from, in the same
    order: `chunks[i]` is the chunk whose vector lives at index position i.
    The two must always move together - this class exists specifically so
    they can never accidentally drift out of sync.
    """

    index: faiss.Index
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index has {self.index.ntotal} vectors but {len(self.chunks)} chunks "
                "were provided - these must stay in lockstep for metadata lookup to be correct."
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a single
        query vector, most similar first. `query_vector` must already be
        embedded the same way the index's vectors were (same model, also
        normalized) - turning question text into a vector is the
        retriever's job (Phase 6), not this module's.
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1  # FAISS pads with -1 if fewer than top_k results exist
        ]

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json())
                f.write("\n")

    @classmethod
    def load(cls, input_dir: Path) -> "VectorStore":
        input_dir = Path(input_dir)
        index_path = input_dir / "index.faiss"
        chunks_path = input_dir / "chunks.jsonl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No persisted vector store found in {input_dir}")

        index = faiss.read_index(str(index_path))
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [Chunk.model_validate(json.loads(line)) for line in f if line.strip()]
        return cls(index=index, chunks=chunks)


def build_vector_store(vectors: np.ndarray, chunks: list[Chunk]) -> VectorStore:
    """Build a fresh in-memory VectorStore from embedding vectors and their
    aligned chunks.

    Vectors are expected to already be L2-normalized (Embedder does this by
    default): we verify that rather than silently re-normalizing, because
    quietly rewriting vectors here would mask a bug introduced upstream.
    """
    if len(vectors) != len(chunks):
        raise ValueError(f"vectors ({len(vectors)}) and chunks ({len(chunks)}) length mismatch")

    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) > 0:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORMALIZATION_TOLERANCE):
            raise ValueError(
                "Vectors are not L2-normalized; build_vector_store expects normalized "
                "embeddings so inner product equals cosine similarity."
            )

    index = faiss.IndexFlatIP(vectors.shape[1])
    if len(vectors) > 0:
        index.add(vectors)

    return VectorStore(index=index, chunks=chunks)
