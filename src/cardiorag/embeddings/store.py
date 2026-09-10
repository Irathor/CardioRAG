"""Persistence for embedding vectors + their aligned chunk metadata.

Not part of the original architecture sketch (which only listed embedder.py)
but split out because it's a distinct concern: embedder.py computes
vectors, this module is responsible for saving/loading them so the corpus
never needs to be re-embedded on every startup or test run.

Two files are always written together for one (corpus, model) pair:
  - "<slug>.npz"   the vector matrix plus the chunk_id order it corresponds to
  - "<slug>.jsonl" the full Chunk metadata, one JSON object per line, in that
                    same order

Keeping chunk_ids inside the .npz lets load_embeddings() verify the two
files still agree on order/count before handing anything to the caller,
instead of silently trusting two separately-written files stayed in sync.
"""

import json
from pathlib import Path

import numpy as np

from cardiorag.embeddings.embedder import EmbeddingResult
from cardiorag.models import Chunk


def model_slug(model_name: str) -> str:
    """Filesystem-safe stand-in for a model name, e.g.
    'sentence-transformers/all-MiniLM-L6-v2' -> 'sentence-transformers__all-MiniLM-L6-v2'."""
    return model_name.replace("/", "__")


def save_embeddings(output_dir: Path, chunks: list[Chunk], result: EmbeddingResult) -> None:
    if len(chunks) != len(result.vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(result.vectors)}) must be the same length"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(result.model_name)

    chunk_ids = np.array([c.chunk_id for c in chunks])
    np.savez_compressed(
        output_dir / f"{slug}.npz",
        vectors=result.vectors,
        chunk_ids=chunk_ids,
        dimension=result.dimension,
        normalized=result.normalized,
    )

    jsonl_path = output_dir / f"{slug}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")


def load_embeddings(output_dir: Path, model_name: str) -> tuple[np.ndarray, list[Chunk]]:
    output_dir = Path(output_dir)
    slug = model_slug(model_name)
    npz_path = output_dir / f"{slug}.npz"
    jsonl_path = output_dir / f"{slug}.jsonl"

    if not npz_path.exists() or not jsonl_path.exists():
        raise FileNotFoundError(
            f"No persisted embeddings for model {model_name!r} in {output_dir}"
        )

    with np.load(npz_path) as data:
        vectors = data["vectors"]
        stored_chunk_ids = list(data["chunk_ids"])

    with open(jsonl_path, encoding="utf-8") as f:
        chunks = [Chunk.model_validate(json.loads(line)) for line in f if line.strip()]

    chunk_ids = [c.chunk_id for c in chunks]
    if chunk_ids != stored_chunk_ids:
        raise ValueError(
            f"Chunk order mismatch between {npz_path.name} and {jsonl_path.name}; "
            "the two files no longer agree on chunk order and cannot be trusted together."
        )
    if len(vectors) != len(chunks):
        raise ValueError(
            f"Vector count ({len(vectors)}) does not match chunk count ({len(chunks)})"
        )

    return vectors, chunks
