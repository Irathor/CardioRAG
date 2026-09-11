"""FastAPI backend (Phase 15) exposing the CardioRAG pipeline over HTTP.

No /evaluate endpoint: retrieval/generation evaluation (Phases 12-13) is a
long-running batch process over dozens of LLM calls, not a request/response
operation - forcing it behind a synchronous HTTP endpoint would either time
out real clients or require background-job infrastructure that doesn't
exist yet. The evaluation scripts remain the right interface for that.
"""

import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from cardiorag.api.auth import require_api_key
from cardiorag.api.dependencies import (
    enforce_rate_limit,
    get_llm_provider,
    get_reranker,
    get_retriever,
    get_vector_store,
)
from cardiorag.api.schemas import (
    CitationWarning,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceInfo,
)
from cardiorag.config import settings
from cardiorag.generation.citations import find_fabricated_quotes
from cardiorag.generation.generator import generate_answer
from cardiorag.generation.providers import LLMProvider
from cardiorag.models import RetrievedChunk
from cardiorag.observability import configure_logging, reset_request_id, set_request_id
from cardiorag.retrieval.hybrid_retriever import HybridRetriever
from cardiorag.retrieval.reranker import Reranker
from cardiorag.retrieval.vector_store import VectorStore

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioRAG API",
    description=(
        "Research/educational RAG system over cardiovascular MRI and AI literature. "
        "NOT a medical diagnostic system; output is not medical advice."
    ),
    version="0.1.0",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assigns one request id per incoming request (Phase 17): every log
    emitted while handling it - including from retrieval/reranking/
    generation modules deep in the call stack - is automatically stamped
    with this id via observability.py's contextvar-backed logging filter,
    without those modules needing to know a request is even happening.
    """
    request_id = str(uuid.uuid4())
    token = set_request_id(request_id)
    start = time.perf_counter()
    try:
        # Every exception type this app can raise has a registered handler
        # (ValueError, NotImplementedError, HTTPException, and a catch-all
        # Exception handler below), so call_next always returns a Response -
        # Starlette resolves exception handlers underneath custom middleware,
        # it doesn't let them propagate out here.
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request completed",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_request_id(token)


def _to_source_info(retrieved: RetrievedChunk) -> SourceInfo:
    chunk = retrieved.chunk
    return SourceInfo(
        title=chunk.title,
        pages=chunk.page_numbers,
        doi=chunk.doi,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        score=retrieved.score,
        text=chunk.text,
    )


@app.get("/health", response_model=HealthResponse)
def health(vector_store: VectorStore = Depends(get_vector_store)) -> HealthResponse:
    """Deliberately exempt from require_api_key/enforce_rate_limit (Phase 20
    fix #6): the Docker Compose healthcheck polls this every 10s with plain
    curl and no credentials, and it exposes nothing more sensitive than
    whether the process is up."""
    llm_configured = (
        (settings.llm_provider == "openai" and bool(settings.openai_api_key))
        or (settings.llm_provider == "groq" and bool(settings.groq_api_key))
    )
    return HealthResponse(
        status="ok",
        index_loaded=True,
        num_chunks=vector_store.index.ntotal,
        llm_provider_configured=llm_configured,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    # Rate limit checked BEFORE the API key (Phase 20 fix #6): otherwise a
    # rejected (401) request never reaches enforce_rate_limit, and an
    # attacker could brute-force the key with unlimited unthrottled guesses.
    # Charging the per-IP budget first closes that gap at the cost of a
    # wrong key also consuming it - an acceptable tradeoff for a
    # single-shared-secret setup.
    _rate_limit: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
) -> RetrieveResponse:
    start = time.perf_counter()
    try:
        results = retriever.retrieve(request.question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    return RetrieveResponse(
        question=request.question,
        results=[_to_source_info(r) for r in results],
        latency_ms=latency_ms,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    provider: LLMProvider = Depends(get_llm_provider),
    _rate_limit: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
) -> QueryResponse:
    start = time.perf_counter()
    try:
        candidates = retriever.retrieve(request.question, top_k=request.retrieve_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rerank_result = reranker.rerank(request.question, candidates, top_k=request.top_k)

    try:
        result = generate_answer(provider, request.question, rerank_result.reranked)
    except Exception as exc:
        logger.exception("Generation failed for question: %r", request.question)
        raise HTTPException(status_code=502, detail="The LLM provider failed to generate an answer.") from exc

    fabricated = find_fabricated_quotes(result.answer, result.sources)
    if fabricated:
        logger.warning(
            "Possible fabricated citation(s) in generated answer",
            extra={"fabricated_quotes": fabricated, "question": request.question},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[_to_source_info(r) for r in result.sources],
        latency_ms=latency_ms,
        citation_warnings=[CitationWarning(**f) for f in fabricated],
    )


@app.get("/documents", response_model=DocumentsResponse)
def documents(
    vector_store: VectorStore = Depends(get_vector_store),
    _rate_limit: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
) -> DocumentsResponse:
    by_document: dict[str, DocumentInfo] = {}
    for chunk in vector_store.chunks:
        if chunk.document_id not in by_document:
            by_document[chunk.document_id] = DocumentInfo(
                document_id=chunk.document_id,
                title=chunk.title,
                doi=chunk.doi,
                filename=chunk.source_filename,
                num_chunks=0,
            )
        by_document[chunk.document_id].num_chunks += 1

    return DocumentsResponse(documents=list(by_document.values()))


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    # Raised by get_llm_provider (Depends) when the configured provider's API
    # key is missing - a server misconfiguration, not a client error nor an
    # unexpected crash, hence 503 rather than 400 or 500.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(NotImplementedError)
def not_implemented_handler(request: Request, exc: NotImplementedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
