import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cardiorag.api.dependencies import (
    get_llm_provider,
    get_rate_limiter,
    get_reranker,
    get_retriever,
    get_vector_store,
)
from cardiorag.api.main import app
from cardiorag.api.rate_limit import RateLimiter
from cardiorag.config import settings
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


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip("\n").split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        event_line = next(line for line in lines if line.startswith("event: "))
        data_line = next(line for line in lines if line.startswith("data: "))
        events.append((event_line[len("event: ") :], json.loads(data_line[len("data: ") :])))
    return events


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

    def stream(self, system_prompt: str, user_prompt: str):
        # Split on spaces (keeping them) so the joined pieces reconstruct
        # generate()'s exact text - both /query and /query/stream should
        # agree on what this fake "answers".
        yield from ["a ", "grounded ", "answer ", "[Source 1]"]


@pytest.fixture
def client(monkeypatch):
    # Auth off and rate limiting effectively unlimited by default: these are
    # covered by their own dedicated tests below, and must not make every
    # other test in this file depend on request count or a clean env.
    monkeypatch.setattr(settings, "api_key", None)

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
    generous_limiter = RateLimiter(max_requests=10_000)
    app.dependency_overrides[get_rate_limiter] = lambda: generous_limiter

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
    assert body["citation_warnings"] == []


def test_query_surfaces_citation_warnings_for_a_fabricated_quote(client):
    class _FabricatingProvider:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return '[Source 1] states: "a completely fabricated sentence not in any real source."'

    app.dependency_overrides[get_llm_provider] = lambda: _FabricatingProvider()

    response = client.post("/query", json={"question": "a real question", "top_k": 1})

    assert response.status_code == 200
    warnings = response.json()["citation_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["source_number"] == 1


def test_query_returns_503_when_llm_provider_misconfigured(client):
    def _raise():
        raise ValueError("GROQ_API_KEY is not set")

    app.dependency_overrides[get_llm_provider] = _raise

    response = client.post("/query", json={"question": "a real question"})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


# --- /query/stream (SSE, project improvement round) ---


def test_query_stream_emits_sources_then_tokens_then_done(client):
    response = client.post("/query/stream", json={"question": "a real question", "top_k": 2})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)

    assert events[0][0] == "sources"
    assert len(events[0][1]["sources"]) == 2

    token_events = [e for e in events if e[0] == "token"]
    assert len(token_events) == 4  # matches _FakeProvider.stream()'s 4 pieces
    reconstructed = "".join(e[1]["text"] for e in token_events)
    assert reconstructed == "a grounded answer [Source 1]"

    assert events[-1][0] == "done"
    assert events[-1][1]["answer"] == "a grounded answer [Source 1]"
    assert events[-1][1]["citation_warnings"] == []
    assert isinstance(events[-1][1]["latency_ms"], int)


def test_query_stream_rejects_empty_question_before_streaming_starts(client):
    response = client.post("/query/stream", json={"question": "   "})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


def test_query_stream_surfaces_citation_warnings_for_a_fabricated_quote(client):
    class _FabricatingStreamingProvider:
        def stream(self, system_prompt: str, user_prompt: str):
            yield '[Source 1] states: "a completely fabricated sentence not in any real source."'

    app.dependency_overrides[get_llm_provider] = lambda: _FabricatingStreamingProvider()

    response = client.post("/query/stream", json={"question": "a real question", "top_k": 1})

    events = _parse_sse(response.text)
    done_event = next(e for e in events if e[0] == "done")
    warnings = done_event[1]["citation_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["source_number"] == 1


def test_query_stream_emits_an_error_event_on_generation_failure(client):
    class _FailingProvider:
        def stream(self, system_prompt: str, user_prompt: str):
            yield "partial "
            raise RuntimeError("boom")

    app.dependency_overrides[get_llm_provider] = lambda: _FailingProvider()

    response = client.post("/query/stream", json={"question": "a real question"})

    assert response.status_code == 200  # already committed by the time generation fails
    events = _parse_sse(response.text)
    assert events[-1][0] == "error"
    assert not any(e[0] == "done" for e in events)


def test_documents_groups_chunks_by_document_id(client):
    response = client.get("/documents")

    assert response.status_code == 200
    documents = {d["document_id"]: d for d in response.json()["documents"]}
    assert documents["docA"]["num_chunks"] == 2
    assert documents["docB"]["num_chunks"] == 1
    assert documents["docA"]["title"] == "A Test Paper"


# --- Fix #6: API-key auth ---


def test_protected_endpoint_rejects_missing_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    response = client.post("/retrieve", json={"question": "a real question"})

    assert response.status_code == 401


def test_protected_endpoint_rejects_wrong_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    response = client.post(
        "/retrieve", json={"question": "a real question"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401


def test_protected_endpoint_accepts_correct_key(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    response = client.post(
        "/retrieve", json={"question": "a real question"}, headers={"X-API-Key": "secret123"}
    )

    assert response.status_code == 200


def test_health_never_requires_a_key_even_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    response = client.get("/health")

    assert response.status_code == 200


# --- Fix #6: rate limiting ---


def test_protected_endpoint_returns_429_once_the_limit_is_exceeded(client):
    # One instance reused across requests, not a fresh one per call - a
    # fresh RateLimiter would never see a prior request and could never
    # trigger 429 in this test.
    limiter = RateLimiter(max_requests=1)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    first = client.get("/documents")
    second = client.get("/documents")

    assert first.status_code == 200
    assert second.status_code == 429


def test_wrong_key_attempts_still_count_against_the_rate_limit(client, monkeypatch):
    """A failed-auth request must still be throttled, or the rate limiter
    does nothing to stop unlimited API-key brute-forcing (Phase 20 fix #6)."""
    monkeypatch.setattr(settings, "api_key", "secret123")
    limiter = RateLimiter(max_requests=1)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    first = client.post("/retrieve", json={"question": "q"}, headers={"X-API-Key": "wrong"})
    second = client.post("/retrieve", json={"question": "q"}, headers={"X-API-Key": "wrong"})

    assert first.status_code == 401
    assert second.status_code == 429


def test_health_is_never_rate_limited(client):
    limiter = RateLimiter(max_requests=1)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200
