"""Request/response models for the REST API.

Deliberate deviation from the master spec's /query example: it shows
`"page": 7` (a single int), but chunks have carried `page_numbers: list[int]`
since Phase 3 - a chunk can legitimately span a page boundary. Collapsing
that to one page here would silently lose real citation information, so
`SourceInfo.pages` is a list, matching what the rest of the system actually
tracks rather than the spec's simplified example.
"""

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    title: str | None = None
    pages: list[int]
    doi: str | None = None
    chunk_id: str
    document_id: str
    score: float
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, gt=0, description="Final number of sources used for generation")
    retrieve_k: int = Field(default=20, gt=0, description="Candidates pulled before reranking")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    latency_ms: int


class RetrieveRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, gt=0)


class RetrieveResponse(BaseModel):
    question: str
    results: list[SourceInfo]
    latency_ms: int


class DocumentInfo(BaseModel):
    document_id: str
    title: str | None = None
    doi: str | None = None
    filename: str
    num_chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: str  # "ok" or "degraded"
    index_loaded: bool
    num_chunks: int
    llm_provider_configured: bool
