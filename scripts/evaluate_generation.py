"""Phase 13: measure generation quality against the real LLM, for all 38
questions in the evaluation dataset - both the 34 answerable ones and the
4 no-evidence ones (which test refusal behavior, not faithfulness).

For each question: retrieve -> rerank -> generate (the real pipeline),
then measure faithfulness, answer relevance, context relevance, citation
correctness (Phase 10), and refusal correctness.

Retrieval failure vs. generation failure are distinguished explicitly: if
none of the retrieved chunks were actually relevant (Phase 12's relevance
criterion), a low faithfulness score means there was nothing faithful to
be had (a RETRIEVAL failure) - not that the model had good evidence and
still misused it (a GENERATION failure).

Checkpointed: results are written to CSV after every question, and a
question already present in that CSV is skipped on the next run. Free-tier
LLM APIs enforce per-model daily token budgets that a ~5-call-per-question
evaluation loop can realistically exhaust mid-run (this happened twice
during development); losing all prior progress on a crash and re-spending
tokens re-answering already-scored questions would be wasteful and, worse,
non-reproducible (a re-asked question can get a different real answer).
"""

import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from cardiorag.config import settings
from cardiorag.embeddings.embedder import Embedder
from cardiorag.evaluation.dataset import load_evaluation_dataset
from cardiorag.evaluation.evaluator import is_relevant_chunk
from cardiorag.evaluation.generation_metrics import (
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
    looks_like_refusal,
)
from cardiorag.generation.citations import find_invalid_citations
from cardiorag.generation.context_builder import build_context
from cardiorag.generation.generator import generate_answer
from cardiorag.generation.providers import RetryingProvider, load_provider_from_settings
from cardiorag.retrieval.reranker import Reranker, retrieve_and_rerank
from cardiorag.retrieval.retriever import load_retriever

logging.basicConfig(level=logging.WARNING)

RESULTS_PATH = Path("data/evaluation/generation_eval_results.csv")


def _load_completed_ids() -> tuple[pd.DataFrame, set[str]]:
    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        return existing, set(existing["id"])
    return pd.DataFrame(), set()


def _evaluate_one(provider, embedder, retriever, reranker, example) -> dict:
    rerank_result = retrieve_and_rerank(retriever, reranker, example.question, retrieve_k=20, final_k=5)
    sources = rerank_result.reranked
    result = generate_answer(provider, example.question, sources)
    context = build_context(sources)

    retrieved_any_relevant = any(is_relevant_chunk(r.chunk, example) for r in sources)

    faithfulness = answer_relevance = context_relevance = None
    if sources and result.answer.strip():
        faithfulness = evaluate_faithfulness(provider, example.question, result.answer, context)
        answer_relevance = evaluate_answer_relevance(provider, embedder, example.question, result.answer)
        context_relevance = evaluate_context_relevance(provider, example.question, context)

    invalid_citations = find_invalid_citations(result.answer, num_sources=len(sources))
    refused = looks_like_refusal(result.answer)
    refusal_correct = (not refused) if example.answerable else refused

    failure_type = None
    if example.answerable and faithfulness is not None and faithfulness.score < 0.5:
        failure_type = "retrieval" if not retrieved_any_relevant else "generation"

    return {
        "id": example.id,
        "category": example.category.value,
        "answerable": example.answerable,
        "faithfulness": faithfulness.score if faithfulness else None,
        "unsupported_claims": len(faithfulness.unsupported_statements) if faithfulness else None,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "invalid_citations": len(invalid_citations),
        "refused": refused,
        "refusal_correct": refusal_correct,
        "retrieved_any_relevant": retrieved_any_relevant,
        "failure_type": failure_type,
    }


def _print_summary(df: pd.DataFrame) -> None:
    answerable_df = df[df["answerable"]]
    no_evidence_df = df[~df["answerable"]]

    print(f"\n=== Aggregate (answerable questions, n={len(answerable_df)}) ===")
    print(f"Mean faithfulness: {answerable_df['faithfulness'].mean():.3f}")
    print(f"Mean answer relevance: {answerable_df['answer_relevance'].mean():.3f}")
    print(f"Mean context relevance: {answerable_df['context_relevance'].mean():.3f}")
    print(f"Total invalid (hallucinated) citations: {answerable_df['invalid_citations'].sum()}")
    print(f"Refusal correctness (should NOT refuse): {answerable_df['refusal_correct'].mean():.1%}")
    print(f"Retrieval-attributed failures: {(answerable_df['failure_type'] == 'retrieval').sum()}")
    print(f"Generation-attributed failures: {(answerable_df['failure_type'] == 'generation').sum()}")

    print(f"\n=== No-evidence questions (n={len(no_evidence_df)}) ===")
    print(f"Refusal correctness (SHOULD refuse): {no_evidence_df['refusal_correct'].mean():.1%}")
    print(f"Total invalid (hallucinated) citations: {no_evidence_df['invalid_citations'].sum()}")


def main() -> None:
    examples = load_evaluation_dataset(Path("data/evaluation/eval_dataset.jsonl"))
    provider = RetryingProvider(load_provider_from_settings(settings))
    embedder = Embedder(settings.embedding_model, device="cpu")
    retriever = load_retriever(Path(settings.index_dir), settings.embedding_model, device="cpu")
    reranker = Reranker(settings.reranker_model, device="cpu")

    existing_df, completed_ids = _load_completed_ids()
    if completed_ids:
        print(f"Resuming: {len(completed_ids)}/{len(examples)} questions already scored, skipping those.")

    remaining = [e for e in examples if e.id not in completed_ids]
    all_rows = existing_df.to_dict("records")

    for i, example in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)} remaining] {example.id}: {example.question[:60]}", flush=True)
        row = _evaluate_one(provider, embedder, retriever, reranker, example)
        all_rows.append(row)

        # Persist after every question so a crash (e.g. a daily token budget
        # cutoff) loses at most the in-flight question, not the whole run.
        pd.DataFrame(all_rows).to_csv(RESULTS_PATH, index=False)
        time.sleep(1.0)  # be polite to the free-tier rate limit

    df = pd.DataFrame(all_rows)
    print("\n" + df.to_string(index=False))
    _print_summary(df)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
