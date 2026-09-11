"""Integration: the complete RAG pipeline, PDF bytes to a cited, grounded
answer - chaining ingestion, cleaning, chunking, embedding, indexing,
retrieval, reranking, generation, and citation verification (Phases 1-10)
through real modules. Only the LLM provider is faked, so the test stays
deterministic and needs no network call/API key - everything else in the
pipeline is the real implementation.
"""

import numpy as np

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.generation.citations import find_invalid_citations
from cardiorag.generation.generator import INSUFFICIENT_EVIDENCE_MESSAGE, generate_answer
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus
from cardiorag.retrieval.reranker import retrieve_and_rerank
from cardiorag.retrieval.retriever import Retriever
from cardiorag.retrieval.vector_store import build_vector_store


class _FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def test_full_pipeline_pdf_to_grounded_cited_answer(synthetic_corpus, embedder, reranker):
    result = load_corpus(synthetic_corpus)
    all_chunks = []
    for doc in result.documents:
        cleaned = clean_document(doc)
        all_chunks.extend(chunk_document(cleaned, ChunkingConfig(chunk_size=64, chunk_overlap=8)))

    vectors = embedder.embed([c.text for c in all_chunks]).vectors
    store = build_vector_store(vectors, all_chunks)
    retriever = Retriever(embedder, store)

    question = "How can deep learning help segment the left ventricle in cardiac MRI?"
    # final_k=1: the synthetic corpus has only 2 chunks total, one per document,
    # so asking for 2 would trivially include the unrelated one too - the real
    # thing to verify is that reranking puts the relevant chunk on top.
    rerank_result = retrieve_and_rerank(retriever, reranker, question, retrieve_k=5, final_k=1)

    assert rerank_result.reranked, "reranking should not discard every candidate"

    provider = _FakeProvider(
        "Deep learning models automate left ventricle segmentation on cardiac MRI [Source 1]."
    )
    answer = generate_answer(provider, question, rerank_result.reranked)

    # The pipeline actually called the LLM with real retrieved context, not an empty prompt.
    assert provider.calls
    _, user_prompt = provider.calls[0]
    assert "[Source 1]" in user_prompt

    # Every source traces back to the relevant paper, never the unrelated one.
    assert answer.sources
    assert all(s.chunk.title == "Deep Learning for Cardiac MRI Segmentation" for s in answer.sources)

    # The answer's citations are all real - Phase 10's checker applied to a real pipeline run.
    assert find_invalid_citations(answer.answer, num_sources=len(answer.sources)) == set()


def test_full_pipeline_refuses_without_calling_llm_when_nothing_retrieved(embedder, reranker):
    """An empty index (e.g. a corpus that produced zero usable chunks) must
    still produce a safe, explicit refusal rather than an empty/broken call."""
    empty_store = build_vector_store(np.empty((0, embedder.dimension), dtype=np.float32), [])
    retriever = Retriever(embedder, empty_store)

    rerank_result = retrieve_and_rerank(
        retriever, reranker, "any question at all", retrieve_k=5, final_k=2
    )
    provider = _FakeProvider("should never be used")
    answer = generate_answer(provider, "any question at all", rerank_result.reranked)

    assert answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert answer.sources == []
    assert provider.calls == []
