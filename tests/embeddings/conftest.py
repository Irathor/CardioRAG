import pytest

from cardiorag.embeddings.embedder import Embedder


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Loaded once per test session - loading a sentence-transformers model
    takes real time, and every test in this file can share one instance."""
    return Embedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
