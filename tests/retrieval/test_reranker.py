import pytest

from cardiorag.models import Chunk, RetrievedChunk
from cardiorag.retrieval.reranker import RerankResult


def _make_retrieved(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        token_count=10,
        page_numbers=[1],
        title="A Test Paper",
        doi=None,
        source_filename="synthetic.pdf",
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_rerank_empty_candidates_returns_empty(reranker):
    result = reranker.rerank("any query", [], top_k=5)

    assert result == RerankResult(reranked=[], rerank_seconds=0.0)


def test_rerank_reduces_to_top_k(reranker):
    candidates = [
        _make_retrieved(f"c{i}", f"unrelated filler text number {i}", score=0.5)
        for i in range(8)
    ]

    result = reranker.rerank("cardiac imaging", candidates, top_k=3)

    assert len(result.reranked) == 3


def test_rerank_returns_fewer_than_top_k_when_candidates_are_scarce(reranker):
    candidates = [_make_retrieved("c0", "some text", score=0.5)]

    result = reranker.rerank("a query", candidates, top_k=5)

    assert len(result.reranked) == 1


def test_rerank_sorts_by_score_descending(reranker):
    candidates = [
        _make_retrieved("relevant", "Cardiac MRI directly measures left ventricular ejection fraction.", 0.5),
        _make_retrieved("irrelevant", "The stock market closed higher today amid trading volume.", 0.5),
    ]

    result = reranker.rerank("What does cardiac MRI measure?", candidates, top_k=2)

    scores = [r.score for r in result.reranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_ranks_substantive_explanation_above_citation_list_entry(reranker):
    # This directly targets the Phase 5/6 finding: dense retrieval sometimes
    # ranked a bibliography entry above real explanatory text because they
    # share vocabulary. A cross-encoder should not make that mistake, since
    # it can judge whether the passage actually answers the question.
    citation_entry = _make_retrieved(
        "citation",
        "[11] Chong JH, Abdulkareem M, Petersen SE, Khanji MY. Artificial "
        "intelligence and cardiovascular magnetic resonance imaging in "
        "myocardial infarction. Eur Heart J 2021.",
        score=0.68,
    )
    substantive_explanation = _make_retrieved(
        "explanation",
        "Artificial intelligence improves cardiac MRI segmentation by "
        "automatically delineating the left and right ventricles on cine "
        "images, reducing analysis time and inter-observer variability.",
        score=0.65,
    )

    result = reranker.rerank(
        "How does artificial intelligence improve cardiac MRI segmentation?",
        [citation_entry, substantive_explanation],
        top_k=2,
    )

    assert result.reranked[0].chunk.chunk_id == "explanation"
