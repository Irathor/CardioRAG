from cardiorag.models import Chunk
from cardiorag.retrieval.bm25_index import BM25Index
from cardiorag.retrieval.hybrid_retriever import HybridRetriever
from cardiorag.retrieval.vector_store import build_vector_store


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        token_count=len(text.split()),
        page_numbers=[1],
        title="A Test Paper",
        doi=None,
        source_filename="synthetic.pdf",
    )


def test_hybrid_retriever_finds_a_rare_exact_term_a_pure_dense_search_might_dilute(embedder):
    # A realistic hybrid-retrieval scenario: a rare, specific identifier
    # (a dataset name) that appears verbatim in exactly one chunk, buried
    # among several chunks that are all semantically about "datasets" in
    # general - dense similarity alone doesn't strongly prefer the exact
    # match over the merely-related ones.
    chunks = [
        _make_chunk("c0", "We evaluated our approach using several benchmark datasets for validation."),
        _make_chunk("c1", "The CardioBenchXYZ99 dataset was used to train and evaluate the model."),
        _make_chunk("c2", "Public datasets are essential for reproducible machine learning research."),
        _make_chunk("c3", "Data collection followed standard protocols across multiple institutions."),
    ]
    vectors = embedder.embed([c.text for c in chunks]).vectors
    vector_store = build_vector_store(vectors, chunks)
    bm25_index = BM25Index(chunks)
    retriever = HybridRetriever(embedder, vector_store, bm25_index, candidate_k=4)

    results = retriever.retrieve("What is CardioBenchXYZ99?", top_k=1)

    assert results[0].chunk.chunk_id == "c1"


def test_hybrid_retriever_respects_top_k(embedder):
    chunks = [_make_chunk(f"c{i}", f"cardiac imaging text number {i}") for i in range(5)]
    vectors = embedder.embed([c.text for c in chunks]).vectors
    vector_store = build_vector_store(vectors, chunks)
    bm25_index = BM25Index(chunks)
    retriever = HybridRetriever(embedder, vector_store, bm25_index, candidate_k=5)

    results = retriever.retrieve("cardiac imaging", top_k=2)

    assert len(results) == 2
