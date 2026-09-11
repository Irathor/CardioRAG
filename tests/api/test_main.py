import numpy as np
import pytest
from fastapi.testclient import TestClient

from cardiorag.api.dependencies import (
    get_llm_provider,
    get_reranker,
    get_retriever,
    get_vector_store,
)
from cardiorag.api.main import app
from cardiorag.models import Chunk, RetrievedChunk
from cardiorag.retrieval.reranker import RerankResult
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import build_vector_store


def _make_chunk(chunk_id: str, document_id: str = "doc1", pages: list[int] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=f"text for {chunk_id}",
        token_count=5,
        page_numbers=pages or [1],
        title="A Test Paper",
        doi="10.1/test",
        source_filename=f"{document_id}.pdf",
    )


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class _FakeEmbedder:
    def __init__(self, dimension: int = 2):
        self.dimension = dimension

    def embed(self, texts: list[str]):
        from types import SimpleNamespace

        # Always "points at" the same direction as the first indexed vector,
        # so retrieve() deterministically returns results in index order.
        vectors = np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))
        return SimpleNamespace(vectors=vectors)


class _FakeReranker:
    """Identity passthrough - returns the candidates unchanged, truncated to top_k."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> RerankResult:
        return RerankResult(reranked=candidates[:top_k], rerank_seconds=0.0)


class _FakeProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "a grounded answer [Source 1]"


@pytest.fixture
def client():
    vector_store = build_vector_store(
        _normalized([[1, 0], [0, 1], [1, 0]]),
        [_make_chunk("c0", "docA", [1]), _make_chunk("c1", "docB", [3]), _make_chunk("c2", "docA", [4, 5])],
    )
    fake_embedder = _FakeEmbedder(dimension=2)
    retriever = Retriever(fake_embedder, vector_store)

    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_reranker] = lambda: _FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: _FakeProvider()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health_reports_index_loaded_and_chunk_count(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["num_chunks"] == 3


def test_retrieve_returns_ranked_sources_with_expected_shape(client):
    response = client.post("/retrieve", json={"question": "a real question", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "a real question"
    assert len(body["results"]) == 2
    result = body["results"][0]
    assert set(result.keys()) == {"title", "pages", "doi", "chunk_id", "document_id", "score", "text"}
    assert isinstance(body["latency_ms"], int)


def test_retrieve_rejects_empty_question(client):
    response = client.post("/retrieve", json={"question": "   "})

    assert response.status_code == 400


def test_query_returns_grounded_answer_with_sources(client):
    response = client.post("/query", json={"question": "a real question", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "a grounded answer [Source 1]"
    assert len(body["sources"]) == 2
    assert isinstance(body["latency_ms"], int)


def test_query_returns_503_when_llm_provider_misconfigured(client):
    def _raise():
        raise ValueError("GROQ_API_KEY is not set")

    app.dependency_overrides[get_llm_provider] = _raise

    response = client.post("/query", json={"question": "a real question"})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_documents_groups_chunks_by_document_id(client):
    response = client.get("/documents")

    assert response.status_code == 200
    documents = {d["document_id"]: d for d in response.json()["documents"]}
    assert documents["docA"]["num_chunks"] == 2
    assert documents["docB"]["num_chunks"] == 1
    assert documents["docA"]["title"] == "A Test Paper"
