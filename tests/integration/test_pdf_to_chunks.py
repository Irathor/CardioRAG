"""Integration: PDF -> chunks, chaining ingestion (Phase 1), cleaning
(Phase 2), and chunking + references-section exclusion (Phase 3 + the
limitation #6 fix) through real modules end to end.
"""

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus


def test_pdf_to_chunks_preserves_content_and_excludes_references(synthetic_corpus):
    result = load_corpus(synthetic_corpus)

    assert result.failures == []
    assert len(result.documents) == 2

    cardiac_doc = next(d for d in result.documents if d.metadata.filename == "cardiac.pdf")
    cleaned = clean_document(cardiac_doc)
    chunks = chunk_document(cleaned, ChunkingConfig(chunk_size=64, chunk_overlap=8))

    assert len(chunks) > 0
    combined_text = " ".join(c.text for c in chunks)

    assert "left ventricle" in combined_text
    assert "myocardial borders" in combined_text
    # The references-section page must never reach chunking.
    assert "Smith J" not in combined_text
    assert "J Cardiovasc Imaging" not in combined_text

    assert all(c.document_id == cardiac_doc.document_id for c in chunks)
    assert all(c.title == "Deep Learning for Cardiac MRI Segmentation" for c in chunks)
    assert all(c.source_filename == "cardiac.pdf" for c in chunks)


def test_pdf_to_chunks_handles_multiple_documents_independently(synthetic_corpus):
    result = load_corpus(synthetic_corpus)

    all_chunks = []
    for doc in result.documents:
        cleaned = clean_document(doc)
        all_chunks.extend(chunk_document(cleaned, ChunkingConfig(chunk_size=64, chunk_overlap=8)))

    document_ids = {c.document_id for c in all_chunks}
    assert len(document_ids) == 2  # one per source PDF, never mixed

    titles = {c.title for c in all_chunks}
    assert titles == {
        "Deep Learning for Cardiac MRI Segmentation",
        "Technology Sector Earnings Report",
    }
