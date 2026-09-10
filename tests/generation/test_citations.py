from cardiorag.generation.citations import (
    build_citation_list,
    extract_cited_source_numbers,
    find_invalid_citations,
    find_uncited_sources,
)
from cardiorag.models import Chunk, RetrievedChunk


def _make_retrieved(chunk_id: str, document_id: str, pages: list[int], score: float = 0.8, **overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id=document_id,
        text="some text",
        token_count=5,
        page_numbers=pages,
        title=f"Paper {document_id}",
        doi=f"10.1/{document_id}",
        source_filename=f"{document_id}.pdf",
    )
    defaults.update(overrides)
    return RetrievedChunk(chunk=Chunk(**defaults), score=score)


# --- extract_cited_source_numbers ---


def test_extract_cited_source_numbers_finds_all_markers():
    text = "LVEF measures function [Source 1]. Segmentation improves accuracy [Source 2]."

    assert extract_cited_source_numbers(text) == {1, 2}


def test_extract_cited_source_numbers_deduplicates_repeated_citations():
    text = "Claim A [Source 1]. Claim B also supported [Source 1]."

    assert extract_cited_source_numbers(text) == {1}


def test_extract_cited_source_numbers_returns_empty_set_when_none_present():
    assert extract_cited_source_numbers("No citations here.") == set()


# --- find_invalid_citations ---


def test_find_invalid_citations_flags_out_of_range_reference():
    text = "This claim cites a source that doesn't exist [Source 5]."

    assert find_invalid_citations(text, num_sources=2) == {5}


def test_find_invalid_citations_empty_when_all_references_are_valid():
    text = "Well supported claim [Source 1] and another [Source 2]."

    assert find_invalid_citations(text, num_sources=2) == set()


def test_find_invalid_citations_flags_zero_or_negative_reference():
    text = "Malformed reference [Source 0]."

    assert find_invalid_citations(text, num_sources=3) == {0}


# --- find_uncited_sources ---


def test_find_uncited_sources_flags_sources_never_referenced():
    text = "Only cites the first one [Source 1]."

    assert find_uncited_sources(text, num_sources=3) == {2, 3}


def test_find_uncited_sources_empty_when_all_cited():
    text = "[Source 1] and [Source 2] both support this."

    assert find_uncited_sources(text, num_sources=2) == set()


# --- build_citation_list ---


def test_build_citation_list_creates_one_citation_per_document():
    sources = [
        _make_retrieved("c0", "docA", [1]),
        _make_retrieved("c1", "docB", [3]),
    ]

    citations = build_citation_list(sources)

    assert [c.document_id for c in citations] == ["docA", "docB"]


def test_build_citation_list_merges_multiple_chunks_from_same_document():
    sources = [
        _make_retrieved("c0", "docA", [1], score=0.9),
        _make_retrieved("c1", "docA", [3, 4], score=0.7),
    ]

    citations = build_citation_list(sources)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.pages == [1, 3, 4]
    assert citation.chunk_ids == ["c0", "c1"]
    assert citation.max_score == 0.9


def test_build_citation_list_preserves_first_appearance_order():
    sources = [
        _make_retrieved("c0", "docB", [1]),
        _make_retrieved("c1", "docA", [1]),
        _make_retrieved("c2", "docB", [2]),
    ]

    citations = build_citation_list(sources)

    assert [c.document_id for c in citations] == ["docB", "docA"]


def test_build_citation_list_empty_for_no_sources():
    assert build_citation_list([]) == []
