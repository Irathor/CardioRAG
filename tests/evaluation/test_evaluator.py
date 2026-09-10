import pytest

from cardiorag.evaluation.evaluator import (
    count_relevant_chunks,
    evaluate_retrieval,
    is_relevant_chunk,
)
from cardiorag.models import Chunk, EvaluationCategory, EvaluationExample, RetrievedChunk


def _chunk(chunk_id: str, document_id: str, pages: list[int]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text="text",
        token_count=5,
        page_numbers=pages,
        title="Title",
        doi=None,
        source_filename=f"{document_id}.pdf",
    )


def _example(**overrides) -> EvaluationExample:
    defaults = dict(
        id="q1",
        question="a question",
        category=EvaluationCategory.FACTUAL,
        expected_document_ids=["docA"],
        expected_pages=[3],
        reference_answer="ref",
        answerable=True,
    )
    defaults.update(overrides)
    return EvaluationExample(**defaults)


def test_is_relevant_chunk_true_when_document_and_page_match():
    chunk = _chunk("c0", "docA", [3, 4])
    assert is_relevant_chunk(chunk, _example()) is True


def test_is_relevant_chunk_false_when_document_matches_but_page_does_not():
    chunk = _chunk("c0", "docA", [10])
    assert is_relevant_chunk(chunk, _example()) is False


def test_is_relevant_chunk_false_when_document_does_not_match():
    chunk = _chunk("c0", "docB", [3])
    assert is_relevant_chunk(chunk, _example()) is False


def test_count_relevant_chunks_counts_across_the_whole_corpus():
    chunks = [_chunk("c0", "docA", [3]), _chunk("c1", "docA", [9]), _chunk("c2", "docB", [3])]
    assert count_relevant_chunks(chunks, _example()) == 1


def test_evaluate_retrieval_raises_when_no_answerable_examples():
    unanswerable = _example(answerable=False, expected_document_ids=[], expected_pages=[])

    with pytest.raises(ValueError, match="answerable"):
        evaluate_retrieval([unanswerable], [], retrieve_fn=lambda q, k: [], k=5)


def test_evaluate_retrieval_perfect_retrieval_gives_perfect_scores():
    example = _example()
    relevant_chunk = _chunk("c0", "docA", [3])
    all_chunks = [relevant_chunk]

    def retrieve_fn(question, k):
        return [RetrievedChunk(chunk=relevant_chunk, score=0.9)]

    result = evaluate_retrieval([example], all_chunks, retrieve_fn, k=5)

    assert result.hit_rate == 1.0
    assert result.mrr == 1.0
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 1.0
    assert result.ndcg_at_k == pytest.approx(1.0)
    assert result.num_questions == 1


def test_evaluate_retrieval_total_miss_gives_zero_scores():
    example = _example()
    relevant_chunk = _chunk("c0", "docA", [3])
    irrelevant_chunk = _chunk("c1", "docB", [1])
    all_chunks = [relevant_chunk]

    def retrieve_fn(question, k):
        return [RetrievedChunk(chunk=irrelevant_chunk, score=0.5)]

    result = evaluate_retrieval([example], all_chunks, retrieve_fn, k=5)

    assert result.hit_rate == 0.0
    assert result.mrr == 0.0
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0


def test_evaluate_retrieval_excludes_unanswerable_examples():
    answerable = _example(id="q1")
    unanswerable = _example(id="q2", answerable=False, expected_document_ids=[], expected_pages=[])
    relevant_chunk = _chunk("c0", "docA", [3])

    def retrieve_fn(question, k):
        return [RetrievedChunk(chunk=relevant_chunk, score=0.9)]

    result = evaluate_retrieval([answerable, unanswerable], [relevant_chunk], retrieve_fn, k=5)

    assert result.num_questions == 1
