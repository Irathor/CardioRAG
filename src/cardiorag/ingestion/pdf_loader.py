"""PDF ingestion: per-page text extraction, document identification, and
detection of common extraction problems.

This module only *extracts* and *flags issues* — it never rewrites text.
Cleaning/normalization happens in ``cleaner.py`` (Phase 2), which can then
decide what to do with each flagged issue while ``raw_text`` stays faithful
to what PyMuPDF actually saw.
"""

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from cardiorag.ingestion.metadata import extract_metadata
from cardiorag.models import (
    Document,
    ExtractionIssue,
    ExtractionIssueType,
    IngestionFailure,
    IngestionResult,
    PageContent,
)

logger = logging.getLogger(__name__)

# Fewer than this many non-whitespace characters on a page counts as effectively empty
# (a bare page number or running header, not real content).
MIN_MEANINGFUL_CHARS = 20
WHITESPACE_RATIO_THRESHOLD = 0.4
# Average line length below this suggests text was extracted one short line at a
# time (typical of a two-column layout read without column awareness).
SHORT_LINE_AVG_THRESHOLD = 40
# A first/last line repeated on at least this many pages is a running header/footer,
# not a coincidence.
HEADER_FOOTER_MIN_REPEATS = 3

HYPHENATION_PATTERN = re.compile(r"[a-z]-\n[a-z]")


def compute_document_id(pdf_bytes: bytes) -> str:
    """Content-addressed ID: identical PDF bytes always yield the same ID
    regardless of filename, so renamed or duplicated files are recognized."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _extract_raw_pages(
    doc: pymupdf.Document, filename: str
) -> tuple[list[str], list[ExtractionIssue]]:
    raw_pages: list[str] = []
    issues: list[ExtractionIssue] = []
    for page_index in range(len(doc)):
        page_number = page_index + 1
        try:
            text = doc[page_index].get_text("text")
        except Exception as exc:  # PyMuPDF can raise on malformed page objects
            logger.warning("Failed to extract page %d of %s: %s", page_number, filename, exc)
            issues.append(
                ExtractionIssue(
                    issue_type=ExtractionIssueType.PAGE_EXTRACTION_ERROR,
                    page_number=page_number,
                    detail=str(exc),
                )
            )
            text = ""
        raw_pages.append(text)
    return raw_pages, issues


def _detect_page_level_issues(raw_pages: list[str]) -> list[ExtractionIssue]:
    issues: list[ExtractionIssue] = []
    for i, text in enumerate(raw_pages):
        page_number = i + 1
        non_ws_chars = len(re.sub(r"\s", "", text))

        if non_ws_chars < MIN_MEANINGFUL_CHARS:
            issues.append(
                ExtractionIssue(
                    issue_type=ExtractionIssueType.EMPTY_PAGE,
                    page_number=page_number,
                    detail=f"{non_ws_chars} non-whitespace characters",
                )
            )
            continue  # other heuristics aren't meaningful on a near-empty page

        if len(text) > 0:
            ws_ratio = 1 - (non_ws_chars / len(text))
            if ws_ratio > WHITESPACE_RATIO_THRESHOLD:
                issues.append(
                    ExtractionIssue(
                        issue_type=ExtractionIssueType.EXCESSIVE_WHITESPACE,
                        page_number=page_number,
                        detail=f"whitespace ratio {ws_ratio:.2f}",
                    )
                )

        hyphen_matches = HYPHENATION_PATTERN.findall(text)
        if hyphen_matches:
            issues.append(
                ExtractionIssue(
                    issue_type=ExtractionIssueType.HYPHENATION_ARTIFACT,
                    page_number=page_number,
                    detail=f"{len(hyphen_matches)} hyphenated line breaks",
                )
            )

        lines = [line for line in text.split("\n") if line.strip()]
        if lines:
            avg_line_len = sum(len(line) for line in lines) / len(lines)
            if avg_line_len < SHORT_LINE_AVG_THRESHOLD:
                issues.append(
                    ExtractionIssue(
                        issue_type=ExtractionIssueType.BROKEN_LINE_WRAPPING,
                        page_number=page_number,
                        detail=f"avg line length {avg_line_len:.1f} chars",
                    )
                )

    return issues


def _detect_repeated_headers_footers(raw_pages: list[str]) -> list[ExtractionIssue]:
    """A running header/footer is a line repeated near-verbatim across many
    pages: the same first (or last) non-blank line on enough pages that it
    can't be coincidence."""
    issues: list[ExtractionIssue] = []
    if len(raw_pages) < HEADER_FOOTER_MIN_REPEATS:
        return issues

    first_lines: dict[str, list[int]] = {}
    last_lines: dict[str, list[int]] = {}
    for i, text in enumerate(raw_pages):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            continue
        first_lines.setdefault(lines[0], []).append(i + 1)
        last_lines.setdefault(lines[-1], []).append(i + 1)

    for line, pages in first_lines.items():
        if len(pages) >= HEADER_FOOTER_MIN_REPEATS:
            for page_number in pages:
                issues.append(
                    ExtractionIssue(
                        issue_type=ExtractionIssueType.DUPLICATED_HEADER,
                        page_number=page_number,
                        detail=line[:80],
                    )
                )

    for line, pages in last_lines.items():
        if len(pages) >= HEADER_FOOTER_MIN_REPEATS:
            for page_number in pages:
                issues.append(
                    ExtractionIssue(
                        issue_type=ExtractionIssueType.DUPLICATED_FOOTER,
                        page_number=page_number,
                        detail=line[:80],
                    )
                )

    return issues


def load_pdf(path: Path) -> Document:
    """Extract a single PDF into a typed ``Document``.

    Raises ``ValueError`` if the file cannot be opened at all. Per-page
    extraction errors do not abort the document: they become
    ``ExtractionIssue`` entries so the failure is visible and traceable
    instead of silently producing an empty page.
    """
    path = Path(path)
    pdf_bytes = path.read_bytes()
    document_id = compute_document_id(pdf_bytes)

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF {path.name}: {exc}") from exc

    try:
        raw_pages, page_errors = _extract_raw_pages(doc, path.name)
        metadata = extract_metadata(doc, path.name, raw_pages)
    finally:
        doc.close()

    pages = [
        PageContent(
            page_number=i + 1,
            raw_text=text,
            char_count=len(text),
            is_empty=len(re.sub(r"\s", "", text)) < MIN_MEANINGFUL_CHARS,
        )
        for i, text in enumerate(raw_pages)
    ]

    issues = (
        page_errors
        + _detect_page_level_issues(raw_pages)
        + _detect_repeated_headers_footers(raw_pages)
    )

    logger.info("Ingested %s: %d pages, %d issues detected", path.name, len(pages), len(issues))

    return Document(document_id=document_id, metadata=metadata, pages=pages, issues=issues)


def load_corpus(corpus_dir: Path) -> IngestionResult:
    """Load every ``*.pdf`` in ``corpus_dir``. A file that fails to open is
    recorded in ``IngestionResult.failures`` (with its filename and error)
    rather than silently dropped or aborting the whole run.
    """
    corpus_dir = Path(corpus_dir)
    result = IngestionResult()

    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        try:
            result.documents.append(load_pdf(pdf_path))
        except ValueError as exc:
            logger.error("Skipping %s: %s", pdf_path.name, exc)
            result.failures.append(IngestionFailure(filename=pdf_path.name, error=str(exc)))

    total_pages = sum(len(d.pages) for d in result.documents)
    issue_counts = Counter(
        issue.issue_type.value for d in result.documents for issue in d.issues
    )
    logger.info(
        "Corpus ingestion complete: %d/%d files ok, %d pages total, issues=%s, failures=%s",
        len(result.documents),
        len(result.documents) + len(result.failures),
        total_pages,
        dict(issue_counts),
        [f.filename for f in result.failures],
    )

    return result
