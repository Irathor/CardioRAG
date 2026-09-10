import pytest

from cardiorag.embeddings.embedder import Embedder
from cardiorag.retrieval.reranker import Reranker


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Loaded once per test session - loading a sentence-transformers model
    takes real time, and every test that needs the real model can share one
    instance instead of reloading it per test file."""
    return Embedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@pytest.fixture(scope="session")
def reranker() -> Reranker:
    return Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
