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


class IngestionFailure(BaseModel):
    filename: str
    error: str


class IngestionResult(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    failures: list[IngestionFailure] = Field(default_factory=list)
