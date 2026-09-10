"""Orchestrates retrieval evaluation: for each answerable question in the
ground-truth dataset, judge which retrieved chunks count as relevant, then
aggregate standard IR metrics across the dataset.
"""

from collections.abc import Callable
from dataclasses import dataclass

from cardiorag.evaluation.retrieval_metrics import (
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from cardiorag.models import Chunk, EvaluationExample, RetrievedChunk


def is_relevant_chunk(chunk: Chunk, example: EvaluationExample) -> bool:
    """A chunk counts as relevant to a question if it comes from one of the
    expected documents AND overlaps at least one expected page.

    Matching on (document, page) rather than an exact chunk_id is
    deliberate: chunk boundaries shift when chunk_size changes (Phase 3),
    so ground truth keyed to one specific chunk_id would silently stop
    matching whenever the chunking configuration under test changes.
    """
    if chunk.document_id not in example.expected_document_ids:
        return False
    return any(p in example.expected_pages for p in chunk.page_numbers)


def count_relevant_chunks(all_chunks: list[Chunk], example: EvaluationExample) -> int:
    return sum(1 for c in all_chunks if is_relevant_chunk(c, example))


@dataclass(frozen=True)
class RetrievalEvalResult:
    hit_rate: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    num_questions: int


def evaluate_retrieval(
    examples: list[EvaluationExample],
    all_chunks: list[Chunk],
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]],
    k: int,
) -> RetrievalEvalResult:
    """Evaluate retrieval quality against every ANSWERABLE example.

    Unanswerable (no_evidence) examples have no relevant chunks to recall
    by construction - they measure refusal behavior, not retrieval, and
    belong to generation evaluation (Phase 13) instead.
    """
    answerable = [e for e in examples if e.answerable]
    if not answerable:
        raise ValueError("No answerable examples to evaluate against")

    hits, rrs, precisions, recalls, ndcgs = [], [], [], [], []
    for example in answerable:
        retrieved = retrieve_fn(example.question, k)
        relevant_flags = [is_relevant_chunk(r.chunk, example) for r in retrieved]
        num_relevant_total = count_relevant_chunks(all_chunks, example)

        hits.append(hit_rate(relevant_flags))
        rrs.append(reciprocal_rank(relevant_flags))
        precisions.append(precision_at_k(relevant_flags))
        recalls.append(recall_at_k(relevant_flags, num_relevant_total))
        if num_relevant_total > 0:
            ndcgs.append(ndcg_at_k(relevant_flags, num_relevant_total))

    return RetrievalEvalResult(
        hit_rate=sum(hits) / len(hits),
        mrr=sum(rrs) / len(rrs),
        precision_at_k=sum(precisions) / len(precisions),
        recall_at_k=sum(recalls) / len(recalls),
        ndcg_at_k=(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        num_questions=len(answerable),
    )
