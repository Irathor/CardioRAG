"""Chunker tests.

Most tests use a fake whitespace tokenizer (deterministic, no network/model
download) to verify the windowing algorithm itself: window size, step,
overlap, page-span attribution. One integration test at the bottom uses the
real embedding model's tokenizer to confirm the algorithm behaves correctly
against actual WordPiece tokenization.
"""

import re

import pytest

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.models import Document, DocumentMetadata, PageContent


class _FakeTokenizer:
    """Whitespace tokenizer: each non-whitespace run is exactly one token.
    Mirrors the real interface (offset_mapping over the original string) so
    chunk_document's windowing logic can be tested without a real model."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "cardiorag.chunking.chunker.get_tokenizer", lambda model_name: _FakeTokenizer()
    )


def _make_document(pages_text: list[str], document_id: str = "doc1") -> Document:
    pages = [
        PageContent(
            page_number=i + 1,
            raw_text=text,
            char_count=len(text),
            is_empty=len(text.strip()) == 0,
            cleaned_text=text,
        )
        for i, text in enumerate(pages_text)
    ]
    metadata = DocumentMetadata(
        filename="synthetic.pdf", title="A Test Paper", doi="10.1/test", num_pages=len(pages)
    )
    return Document(document_id=document_id, metadata=metadata, pages=pages)


def test_chunking_config_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_chunking_config_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0, chunk_overlap=0)


def test_chunk_document_returns_empty_list_for_blank_document(fake_tokenizer):
    document = _make_document(["", "   "])

    assert chunk_document(document) == []


def test_chunk_document_respects_chunk_size_and_produces_overlap(fake_tokenizer):
    words = " ".join(f"w{i}" for i in range(30))  # 30 tokens under the fake tokenizer
    document = _make_document([words])
    config = ChunkingConfig(chunk_size=10, chunk_overlap=3)

    chunks = chunk_document(document, config)

    # windows: [0:10] [7:17] [14:24] [21:30] -> 4 chunks, none exceeding the budget
    assert len(chunks) == 4
    assert all(c.token_count <= 10 for c in chunks)
    # consecutive chunks share the 3-token overlap
    assert "w9" in chunks[0].text and "w9" in chunks[1].text
    assert "w0" not in chunks[1].text


def test_chunk_document_assigns_sequential_ids_and_shared_metadata(fake_tokenizer):
    document = _make_document(["one two three four five"])
    config = ChunkingConfig(chunk_size=2, chunk_overlap=0)

    chunks = chunk_document(document, config)

    assert [c.chunk_id for c in chunks] == [f"{document.document_id}-000{i}" for i in range(3)]
    for c in chunks:
        assert c.document_id == document.document_id
        assert c.title == "A Test Paper"
        assert c.doi == "10.1/test"
        assert c.source_filename == "synthetic.pdf"


def test_chunk_document_tracks_pages_spanned_by_each_chunk(fake_tokenizer):
    page1 = " ".join(f"w{i}" for i in range(10))  # tokens 0-9
    page2 = " ".join(f"w{i}" for i in range(10, 20))  # tokens 10-19
    document = _make_document([page1, page2])
    config = ChunkingConfig(chunk_size=6, chunk_overlap=2)

    chunks = chunk_document(document, config)

    assert chunks[0].page_numbers == [1]  # fully inside page 1
    assert [1, 2] in [c.page_numbers for c in chunks]  # some chunk spans the boundary
    assert chunks[-1].page_numbers == [2]  # fully inside page 2


def test_chunk_document_falls_back_to_raw_text_when_uncleaned(fake_tokenizer):
    page = PageContent(
        page_number=1, raw_text="alpha beta gamma", char_count=17, is_empty=False
    )  # cleaned_text left as None
    metadata = DocumentMetadata(filename="f.pdf", num_pages=1)
    document = Document(document_id="doc2", metadata=metadata, pages=[page])

    chunks = chunk_document(document, ChunkingConfig(chunk_size=10, chunk_overlap=0))

    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma"


def test_chunk_document_skips_pages_marked_as_references(fake_tokenizer):
    body_page = PageContent(
        page_number=1, raw_text="w0 w1 w2 w3", char_count=11, is_empty=False,
        cleaned_text="w0 w1 w2 w3",
    )
    references_page = PageContent(
        page_number=2, raw_text="w4 w5 w6 w7", char_count=11, is_empty=False,
        cleaned_text="w4 w5 w6 w7", is_references_section=True,
    )
    metadata = DocumentMetadata(filename="f.pdf", num_pages=2)
    document = Document(document_id="doc3", metadata=metadata, pages=[body_page, references_page])

    chunks = chunk_document(document, ChunkingConfig(chunk_size=10, chunk_overlap=0))

    assert len(chunks) == 1
    assert "w4" not in chunks[0].text
    assert 2 not in chunks[0].page_numbers


# --- Integration test against the real embedding-model tokenizer (no fake_tokenizer fixture) ---


def test_chunk_document_with_real_tokenizer_stays_within_token_budget():
    document = _make_document(
        [
            "Cardiovascular magnetic resonance (CMR) with T1-weighted imaging provides "
            "accurate and reproducible measurements of myocardial function. " * 15
        ]
    )
    config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)
