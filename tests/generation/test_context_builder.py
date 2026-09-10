from cardiorag.generation.context_builder import build_context
from cardiorag.models import Chunk, RetrievedChunk


def _make_retrieved(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id="c0",
        document_id="doc1",
        text="Cardiac MRI provides high spatial resolution.",
        token_count=8,
        page_numbers=[4, 5],
        title="A Test Paper",
        doi="10.1/test",
        source_filename="synthetic.pdf",
    )
    defaults.update(overrides)
    score = defaults.pop("score", 0.8)
    return RetrievedChunk(chunk=Chunk(**defaults), score=score)


def test_build_context_empty_list_returns_empty_string():
    assert build_context([]) == ""


def test_build_context_includes_all_required_fields():
    context = build_context([_make_retrieved()])

    assert "[Source 1]" in context
    assert "Title: A Test Paper" in context
    assert "Page(s): 4, 5" in context
    assert "DOI: 10.1/test" in context
    assert "Text: Cardiac MRI provides high spatial resolution." in context


def test_build_context_numbers_sources_sequentially():
    context = build_context([_make_retrieved(chunk_id="c0"), _make_retrieved(chunk_id="c1")])

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert context.index("[Source 1]") < context.index("[Source 2]")


def test_build_context_handles_missing_title_and_doi():
    context = build_context([_make_retrieved(title=None, doi=None)])

    assert "Title: Unknown" in context
    assert "DOI: N/A" in context
    assert "None" not in context
