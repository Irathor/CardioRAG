"""Shared typed domain models used across ingestion, chunking, and retrieval.

Keeping these in one place means every later phase (chunking, retrieval,
citations) works against the same contract instead of ad-hoc dicts.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ExtractionIssueType(str, Enum):
    EMPTY_PAGE = "empty_page"
    DUPLICATED_HEADER = "duplicated_header"
    DUPLICATED_FOOTER = "duplicated_footer"
    BROKEN_LINE_WRAPPING = "broken_line_wrapping"
    EXCESSIVE_WHITESPACE = "excessive_whitespace"
    HYPHENATION_ARTIFACT = "hyphenation_artifact"
    PAGE_EXTRACTION_ERROR = "page_extraction_error"


class ExtractionIssue(BaseModel):
    issue_type: ExtractionIssueType
    page_number: int  # 1-indexed, matches how a human would cite the page
    detail: str = ""


class PageContent(BaseModel):
    page_number: int  # 1-indexed
    raw_text: str
    char_count: int
    is_empty: bool
    # Populated by cleaner.clean_document (Phase 2). raw_text is never
    # overwritten, so the original extraction is always available for
    # comparison/debugging.
    cleaned_text: str | None = None
    # True for this page and every later page once a standalone "References"
    # (or "Bibliography") heading line is found. Chunking skips these pages
    # entirely - a bibliography entry sharing vocabulary with a question is
    # not evidence, and indexing it just competes with real evidence at
    # retrieval time.
    is_references_section: bool = False


class DocumentMetadata(BaseModel):
    filename: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    publication_year: int | None = None
    num_pages: int


class Document(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    pages: list[PageContent]
    issues: list[ExtractionIssue] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    page_numbers: list[int]  # 1+ pages this chunk's text was drawn from, sorted
    title: str | None = None
    doi: str | None = None
    source_filename: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class EvaluationCategory(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"
    MULTI_PAPER = "multi_paper"
    NO_EVIDENCE = "no_evidence"
    MISLEADING = "misleading"


class EvaluationExample(BaseModel):
    id: str
    question: str
    category: EvaluationCategory
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    reference_answer: str
    answerable: bool


class Citation(BaseModel):
    """A presentation-ready citation for one document, merging every chunk
    from that document that contributed to an answer. Built strictly from
    RetrievedChunk metadata - never from the LLM's free-text output."""

    document_id: str
    title: str | None = None
    doi: str | None = None
    source_filename: str
    pages: list[int]
    chunk_ids: list[str]
    max_score: float


class IngestionFailure(BaseModel):
    filename: str
    error: str


class IngestionResult(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    failures: list[IngestionFailure] = Field(default_factory=list)
