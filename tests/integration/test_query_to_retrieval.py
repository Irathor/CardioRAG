"""Integration: a natural-language query -> ranked retrieval results,
chaining the full ingest -> clean -> chunk -> embed -> index -> retrieve
path (Phases 1-6) through real modules, using the real embedding model.
"""

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import build_vector_store


def _build_retriever(corpus_dir, embedder) -> Retriever:
    result = load_corpus(corpus_dir)
    all_chunks = []
    for doc in result.documents:
        cleaned = clean_document(doc)
        all_chunks.extend(chunk_document(cleaned, ChunkingConfig(chunk_size=64, chunk_overlap=8)))

    vectors = embedder.embed([c.text for c in all_chunks]).vectors
    store = build_vector_store(vectors, all_chunks)
    return Retriever(embedder, store)


def test_query_to_retrieval_ranks_the_relevant_paper_first(synthetic_corpus, embedder):
    retriever = _build_retriever(synthetic_corpus, embedder)

    results = retriever.retrieve(
        "What deep learning technique delineates myocardial borders on cine MRI?", top_k=2
    )

    assert len(results) == 2
    assert results[0].chunk.title == "Deep Learning for Cardiac MRI Segmentation"
    assert results[0].score > results[1].score

    for result in results:
        assert result.chunk.chunk_id
        assert result.chunk.document_id
        assert result.chunk.page_numbers
        assert isinstance(result.score, float)


def test_query_to_retrieval_respects_top_k(synthetic_corpus, embedder):
    retriever = _build_retriever(synthetic_corpus, embedder)

    results = retriever.retrieve("cardiac imaging", top_k=1)

    assert len(results) == 1
