"""Embedding abstraction: turns chunk/query text into dense vectors, hiding
which concrete model or backend produced them so retrieval code never
depends on a specific embedding model.

Vectors are L2-normalized by default. This is what lets the vector index
(Phase 5) use inner product instead of computing cosine similarity directly:
for unit vectors, inner product *is* cosine similarity, and FAISS's
inner-product index (IndexFlatIP) is faster than a cosine-specific one.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape (n, dimension), float32
    model_name: str
    dimension: int
    normalized: bool
    encode_seconds: float


def resolve_device(device: str) -> str:
    """"auto" picks CUDA when available, otherwise CPU. An explicit value
    (e.g. "cpu") is passed through unchanged so tests and constrained
    environments can force a specific device."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Wraps a sentence-transformers model behind a stable interface.

    Swapping `model_name` (e.g. general-purpose <-> scientific/biomedical)
    changes nothing else in the pipeline: dimension, batching, and
    normalization are all handled here and reported back in the result.
    """

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True):
        resolved_device = resolve_device(device)
        logger.info("Loading embedding model %r on device=%s", model_name, resolved_device)
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self.model_name = model_name
        self.device = resolved_device
        self.normalize = normalize
        self.dimension: int = self._model.get_embedding_dimension()

    def embed(
        self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> EmbeddingResult:
        """Embed `texts` in batches. Returns an empty (0, dimension) array
        for an empty input rather than erroring - a document that produced
        zero chunks is a legitimate, if unhelpful, case."""
        if not texts:
            return EmbeddingResult(
                vectors=np.empty((0, self.dimension), dtype=np.float32),
                model_name=self.model_name,
                dimension=self.dimension,
                normalized=self.normalize,
                encode_seconds=0.0,
            )

        start = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        ).astype(np.float32)
        encode_seconds = time.perf_counter() - start

        logger.info(
            "Embedded %d texts with %s in %.2fs (%.1f texts/s)",
            len(texts),
            self.model_name,
            encode_seconds,
            len(texts) / encode_seconds if encode_seconds > 0 else float("inf"),
        )

        return EmbeddingResult(
            vectors=vectors,
            model_name=self.model_name,
            dimension=self.dimension,
            normalized=self.normalize,
            encode_seconds=encode_seconds,
        )
