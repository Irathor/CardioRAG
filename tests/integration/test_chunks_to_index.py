"""Integration: chunks -> vector index, chaining real embedding (Phase 4)
and FAISS indexing (Phase 5). Uses the real sentence-transformers model
(session-scoped `embedder` fixture from tests/conftest.py) to verify
genuine semantic search behavior, not just structural wiring.
"""

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus
from cardiorag.retrieval.vector_store import build_vector_store


def _index_synthetic_corpus(corpus_dir, embedder):
    result = load_corpus(corpus_dir)
    all_chunks = []
    for doc in result.documents:
        cleaned = clean_document(doc)
        all_chunks.extend(chunk_document(cleaned, ChunkingConfig(chunk_size=64, chunk_overlap=8)))

    vectors = embedder.embed([c.text for c in all_chunks]).vectors
    return build_vector_store(vectors, all_chunks)


def test_chunks_to_index_semantic_search_finds_the_relevant_document(synthetic_corpus, embedder):
    store = _index_synthetic_corpus(synthetic_corpus, embedder)

    query_vector = embedder.embed(
        ["How is artificial intelligence used to segment the heart's left ventricle?"]
    ).vectors[0]
    results = store.search(query_vector, top_k=1)

    assert results[0][0].title == "Deep Learning for Cardiac MRI Segmentation"


def test_chunks_to_index_distinguishes_unrelated_documents(synthetic_corpus, embedder):
    store = _index_synthetic_corpus(synthetic_corpus, embedder)

    query_vector = embedder.embed(
        ["What drove growth in technology sector earnings this quarter?"]
    ).vectors[0]
    results = store.search(query_vector, top_k=1)

    assert results[0][0].title == "Technology Sector Earnings Report"
