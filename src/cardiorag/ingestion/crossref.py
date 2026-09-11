"""Optional metadata enrichment via the CrossRef REST API.

PDF-embedded metadata and heuristic extraction (metadata.py) are often
wrong for scientific papers - title truncation, a citation year mistaken
for the publication year, an incomplete author list (see README
Limitations). The DOI extraction in metadata.py is reliable, though, and
CrossRef's free public API returns authoritative title/author/year data
for a DOI.

This is a deliberately SEPARATE, optional enrichment step - never baked
into load_pdf()/extract_metadata() - so core ingestion stays offline,
deterministic, and fast, which most of this project's ingestion tests
depend on. Call enrich_metadata_from_crossref() explicitly when network
access and better metadata are both wanted.
"""

import logging

import httpx

from cardiorag.models import DocumentMetadata

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works/{doi}"
DEFAULT_TIMEOUT_SECONDS = 5.0


def fetch_crossref_work(doi: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict | None:
    """Query CrossRef for a DOI's metadata. Returns None on any failure
    (network error, timeout, 404, malformed response) rather than raising -
    this is best-effort enrichment, and its absence must never break
    ingestion for callers that don't check the return value.
    """
    try:
        response = httpx.get(
            CROSSREF_API_URL.format(doi=doi),
            timeout=timeout,
            # CrossRef's "polite pool" gives better rate limits to requests
            # that identify themselves - a courtesy, not an API key.
            headers={"User-Agent": "CardioRAG/0.1 (https://github.com/; mailto:noreply@example.com)"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("CrossRef lookup failed for DOI %r: %s", doi, exc)
        return None

    try:
        return response.json()["message"]
    except (KeyError, ValueError) as exc:
        logger.warning("CrossRef returned an unparseable response for DOI %r: %s", doi, exc)
        return None


def _extract_title(work: dict) -> str | None:
    titles = work.get("title") or []
    return titles[0] if titles else None


def _extract_authors(work: dict) -> list[str]:
    authors = []
    for author in work.get("author", []) or []:
        name = f"{author.get('given', '')} {author.get('family', '')}".strip()
        if name:
            authors.append(name)
    return authors


def _extract_year(work: dict) -> int | None:
    for date_field in ("published-print", "published-online", "published", "issued"):
        date_parts = (work.get(date_field) or {}).get("date-parts")
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


def enrich_metadata_from_crossref(
    metadata: DocumentMetadata, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> DocumentMetadata:
    """Return a new DocumentMetadata with title/authors/publication_year
    replaced by CrossRef's data when available, otherwise returned
    unchanged. Requires `metadata.doi` to already be set (from PDF
    extraction) - this looks UP a DOI, it never guesses or searches for one.
    """
    if not metadata.doi:
        return metadata

    work = fetch_crossref_work(metadata.doi, timeout=timeout)
    if work is None:
        return metadata

    title = _extract_title(work) or metadata.title
    authors = _extract_authors(work) or metadata.authors
    year = _extract_year(work) or metadata.publication_year

    return metadata.model_copy(update={"title": title, "authors": authors, "publication_year": year})
