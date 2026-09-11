"""Query-embedding cache (project improvement round, post-Phase 20).

Wraps an `Embedder` and caches the vector for a single-text call, keyed on
normalized query text. Deliberately narrow: it only caches single-text
calls, since that's the only shape `Retriever.retrieve()` ever makes
(`self._embedder.embed([query])`) - a batch call (chunk embedding at
ingestion/index-build time) never repeats the same text within a run and
passes straight through untouched, so this adds zero overhead there.

The real payoff is a long-lived API process seeing the same or a repeated
question from different users/UI sessions - a fresh Python process (a
one-off script, a test run) starts with an empty cache and gets no benefit,
which is expected and fine.
"""

from collections import OrderedDict

import numpy as np

from cardiorag.embeddings.embedder import Embedder, EmbeddingResult


def _normalize(text: str) -> str:
    return text.strip().lower()


class CachingEmbedder:
    def __init__(self, embedder: Embedder, maxsize: int = 256):
        self._embedder = embedder
        self._maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.model_name = embedder.model_name
        self.dimension = embedder.dimension
        self.normalize = embedder.normalize
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        if len(texts) != 1:
            return self._embedder.embed(texts, **kwargs)

        key = _normalize(texts[0])
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)  # LRU: mark as most recently used
            return EmbeddingResult(
                vectors=cached.reshape(1, -1),
                model_name=self.model_name,
                dimension=self.dimension,
                normalized=self.normalize,
                encode_seconds=0.0,
            )

        self.misses += 1
        result = self._embedder.embed(texts, **kwargs)
        self._cache[key] = result.vectors[0].copy()
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict the least recently used
        return result

    def cache_info(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache)}
