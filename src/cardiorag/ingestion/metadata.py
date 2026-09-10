"""Best-effort extraction of bibliographic metadata (title, authors, DOI, year).

Embedded PDF metadata (``doc.metadata``) is frequently missing or useless for
scientific papers: many publisher pipelines leave the title blank or fill it
with a template string. We use the embedded value when it looks real, and
otherwise fall back to regex heuristics over the first couple of pages.

Every field is optional (`None` / empty list) when it cannot be determined —
we never fabricate a title, author, or DOI.
"""

import re

import pymupdf

from cardiorag.models import DocumentMetadata

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Substrings publishers commonly leave in the embedded 'title' field instead of a real title.
_JUNK_TITLE_MARKERS = ("untitled", "microsoft word")


def _looks_like_junk_title(title: str | None) -> bool:
    if not title or not title.strip():
        return True
    lowered = title.strip().lower()
    return any(marker in lowered for marker in _JUNK_TITLE_MARKERS)


def _guess_title_from_text(first_page_text: str) -> str | None:
    """Fall back to the first substantial line of page 1 — in most papers that's
    the title, since running headers/DOIs are usually shorter or matched below."""
    for line in first_page_text.split("\n"):
        candidate = line.strip()
        if len(candidate) >= 15 and not DOI_PATTERN.search(candidate):
            return candidate
    return None


def _find_doi(pages_text: list[str]) -> str | None:
    for text in pages_text:
        match = DOI_PATTERN.search(text)
        if match:
            return match.group(0).rstrip(".,);")
    return None


def _find_year(pages_text: list[str]) -> int | None:
    for text in pages_text:
        match = YEAR_PATTERN.search(text)
        if match:
            return int(match.group(0))
    return None


def _parse_authors(raw_author_field: str | None) -> list[str]:
    """Split an embedded author string into individual names.

    Author formatting is inconsistent across PDF producers, so we pick the
    least ambiguous separator available: semicolons unambiguously separate
    people ("Smith, John; Doe, Jane"); otherwise fall back to " and " or a
    bare comma. A comma-separated "Last, First, Last, First" list without
    semicolons is a known ambiguous case we don't attempt to resolve.
    """
    if not raw_author_field or not raw_author_field.strip():
        return []
    raw = raw_author_field.strip()
    if ";" in raw:
        parts = raw.split(";")
    elif " and " in raw:
        parts = raw.split(" and ")
    else:
        parts = raw.split(",")
    return [p.strip() for p in parts if p.strip()]


def extract_metadata(
    doc: pymupdf.Document, filename: str, raw_pages: list[str]
) -> DocumentMetadata:
    embedded = doc.metadata or {}
    first_pages = raw_pages[:2]

    title = embedded.get("title")
    if _looks_like_junk_title(title):
        title = _guess_title_from_text(raw_pages[0]) if raw_pages else None

    return DocumentMetadata(
        filename=filename,
        title=title,
        authors=_parse_authors(embedded.get("author")),
        doi=_find_doi(first_pages),
        publication_year=_find_year(first_pages),
        num_pages=len(raw_pages),
    )
