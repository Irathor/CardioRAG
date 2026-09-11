import numpy as np

from cardiorag.embeddings.caching_embedder import CachingEmbedder
from cardiorag.embeddings.embedder import EmbeddingResult


class _FakeEmbedder:
    """Counts real calls, so tests can prove the cache actually avoided one."""

    model_name = "fake-model"
    dimension = 3
    normalize = True

    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        self.calls += 1
        vectors = np.array([[len(t), 0.0, 1.0] for t in texts], dtype=np.float32)
        return EmbeddingResult(
            vectors=vectors, model_name=self.model_name, dimension=3, normalized=True, encode_seconds=0.001
        )


def test_repeated_query_hits_the_cache_not_the_real_embedder():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake)

    first = cache.embed(["What is MOCOnet?"])
    second = cache.embed(["What is MOCOnet?"])

    assert fake.calls == 1  # the real embedder was only ever called once
    assert np.array_equal(first.vectors, second.vectors)
    assert cache.cache_info() == {"hits": 1, "misses": 1, "size": 1}


def test_cache_key_normalizes_whitespace_and_case():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake)

    cache.embed(["What is MOCOnet?"])
    cache.embed(["  what is moconet?  "])

    assert fake.calls == 1
    assert cache.cache_info()["hits"] == 1


def test_different_queries_are_not_conflated():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake)

    cache.embed(["question one"])
    cache.embed(["a different, longer question"])

    assert fake.calls == 2
    assert cache.cache_info() == {"hits": 0, "misses": 2, "size": 2}


def test_batch_calls_bypass_the_cache_entirely():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake)

    cache.embed(["chunk text one", "chunk text two"])
    cache.embed(["chunk text one", "chunk text two"])

    assert fake.calls == 2  # never cached - not the shape retrieve() ever uses
    assert cache.cache_info() == {"hits": 0, "misses": 0, "size": 0}


def test_lru_eviction_drops_the_least_recently_used_entry():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake, maxsize=2)

    cache.embed(["q1"])
    cache.embed(["q2"])
    cache.embed(["q1"])  # refreshes q1's recency - q2 is now the LRU entry
    cache.embed(["q3"])  # should evict q2, not q1

    cache.embed(["q1"])
    cache.embed(["q2"])

    assert fake.calls == 4  # q1 miss, q2 miss, q3 miss, q2 re-miss after eviction
    assert cache.cache_info()["size"] == 2


def test_exposes_the_wrapped_embedders_model_name_and_dimension():
    fake = _FakeEmbedder()
    cache = CachingEmbedder(fake)

    assert cache.model_name == "fake-model"
    assert cache.dimension == 3
