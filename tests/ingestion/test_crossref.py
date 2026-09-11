"""Unit tests for the optional CrossRef enrichment. All httpx calls are
mocked - deterministic and offline, consistent with the rest of the
ingestion test suite. See scripts/enrich_metadata.py for a real,
live-network verification against this project's actual corpus DOIs.
"""

import httpx

from cardiorag.ingestion.crossref import enrich_metadata_from_crossref, fetch_crossref_work
from cardiorag.models import DocumentMetadata

_REAL_SHAPED_RESPONSE = {
    "message": {
        "title": ["A Real Paper Title From CrossRef"],
        "author": [
            {"given": "Qiang", "family": "Zhang"},
            {"given": "Anastasia", "family": "Fotaki"},
        ],
        "published-print": {"date-parts": [[2024]]},
    }
}


def _mock_response(json_body: dict, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1/test")
    return httpx.Response(status_code, json=json_body, request=request)


def test_fetch_crossref_work_returns_the_message_payload(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _mock_response(_REAL_SHAPED_RESPONSE))

    work = fetch_crossref_work("10.1/test")

    assert work["title"] == ["A Real Paper Title From CrossRef"]


def test_fetch_crossref_work_returns_none_on_http_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("httpx.get", _raise)

    assert fetch_crossref_work("10.1/test") is None


def test_fetch_crossref_work_returns_none_on_404(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _mock_response({}, status_code=404))

    assert fetch_crossref_work("10.1/nonexistent") is None


def test_fetch_crossref_work_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _mock_response({"unexpected": "shape"}))

    assert fetch_crossref_work("10.1/test") is None


def test_enrich_metadata_replaces_title_authors_and_year(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _mock_response(_REAL_SHAPED_RESPONSE))
    original = DocumentMetadata(
        filename="paper.pdf",
        title="Retrieval-Augmented Generation for",  # truncated, PDF-derived
        authors=["Zhang Q"],
        doi="10.1/test",
        publication_year=2005,  # wrong, PDF-derived
        num_pages=10,
    )

    enriched = enrich_metadata_from_crossref(original)

    assert enriched.title == "A Real Paper Title From CrossRef"
    assert enriched.authors == ["Qiang Zhang", "Anastasia Fotaki"]
    assert enriched.publication_year == 2024


def test_enrich_metadata_skips_lookup_when_no_doi():
    original = DocumentMetadata(filename="paper.pdf", doi=None, num_pages=5)

    enriched = enrich_metadata_from_crossref(original)

    assert enriched == original


def test_enrich_metadata_falls_back_to_original_when_crossref_lookup_fails(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    original = DocumentMetadata(
        filename="paper.pdf", title="Original Title", doi="10.1/test", num_pages=5
    )

    enriched = enrich_metadata_from_crossref(original)

    assert enriched == original


def test_enrich_metadata_keeps_original_fields_crossref_did_not_provide(monkeypatch):
    # CrossRef has a title but no author list for this (real, common) case.
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _mock_response({"message": {"title": ["New Title"]}}),
    )
    original = DocumentMetadata(
        filename="paper.pdf",
        title="Old Title",
        authors=["Original Author"],
        doi="10.1/test",
        publication_year=2020,
        num_pages=5,
    )

    enriched = enrich_metadata_from_crossref(original)

    assert enriched.title == "New Title"
    assert enriched.authors == ["Original Author"]  # kept, CrossRef gave none
    assert enriched.publication_year == 2020  # kept, CrossRef gave none
