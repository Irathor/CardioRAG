"""Phase 14: backfill the experiment log from experiments already run in
Phases 12-13, rather than re-running anything.

This is a one-time formalization step: the numbers below are read directly
from data/evaluation/retrieval_eval_results.csv and
generation_eval_results.csv (both already committed, both produced by
actually running scripts/evaluate_retrieval.py and
scripts/evaluate_generation.py against the real corpus/LLM) plus the fixed
baseline parameters those scripts used internally as constants
(DEFAULT_CHUNK_SIZE=512, DEFAULT_MODEL=MiniLM, DEFAULT_K=5 in
evaluate_retrieval.py) - nothing here is a new measurement.

The two generation-evaluation rows are logged SEPARATELY rather than as one
blended experiment: the 38-question run actually used two different judge
models (qwen/qwen3.8-27b for the first 28 questions, allam-2-7b for the
last 10 after every higher-quality model's free-tier daily quota was
exhausted), and allam-2-7b did not reliably follow the JSON-output format
the judge prompts require - averaging them into one number would hide
that the two subsets are not comparable measurements.
"""

import logging
from pathlib import Path

import pandas as pd

from cardiorag.evaluation.experiment_tracking import log_experiment
from cardiorag.models import ExperimentRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS_PATH = Path("data/evaluation/experiments.jsonl")
DATE = "2026-09-10"  # the real date these experiments were run (see git log)

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER = "sentence-transformers/allenai-specter"


def _retrieval_records() -> list[ExperimentRecord]:
    df = pd.read_csv("data/evaluation/retrieval_eval_results.csv")
    records = []
    i = 0

    def metrics(row) -> dict[str, float]:
        return {
            "hit_rate": row["hit_rate"],
            "mrr": row["mrr"],
            "precision_at_k": row["precision_at_k"],
            "recall_at_k": row["recall_at_k"],
            "ndcg_at_k": row["ndcg_at_k"],
        }

    for _, row in df[df["experiment"] == "chunk_size"].iterrows():
        i += 1
        chunk_size = int(row["variant"])
        records.append(
            ExperimentRecord(
                experiment_id=f"EXP-{i:03d}",
                date=DATE,
                description=f"Phase 12 chunk_size sweep: {chunk_size} tokens",
                embedding_model=MINILM,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 8,
                retrieval_top_k=5,
                reranking_enabled=False,
                final_context_size=5,
                retrieval_metrics=metrics(row),
                notes="Fixed: MiniLM, top_k=5, no reranking. Varying: chunk_size.",
            )
        )

    for _, row in df[df["experiment"] == "embedding_model"].iterrows():
        i += 1
        is_specter = "SPECTER" in row["variant"]
        records.append(
            ExperimentRecord(
                experiment_id=f"EXP-{i:03d}",
                date=DATE,
                description=f"Phase 12 embedding model comparison: {row['variant']}",
                embedding_model=SPECTER if is_specter else MINILM,
                chunk_size=512,
                chunk_overlap=64,
                retrieval_top_k=5,
                reranking_enabled=False,
                final_context_size=5,
                retrieval_metrics=metrics(row),
                notes="Fixed: chunk_size=512, top_k=5, no reranking. Varying: embedding model.",
            )
        )

    for _, row in df[df["experiment"] == "top_k"].iterrows():
        i += 1
        k = int(row["variant"].split("=")[1])
        records.append(
            ExperimentRecord(
                experiment_id=f"EXP-{i:03d}",
                date=DATE,
                description=f"Phase 12 top_k sweep: k={k}",
                embedding_model=MINILM,
                chunk_size=512,
                chunk_overlap=64,
                retrieval_top_k=k,
                reranking_enabled=False,
                final_context_size=k,
                retrieval_metrics=metrics(row),
                notes="Fixed: chunk_size=512, MiniLM, no reranking. Varying: top_k.",
            )
        )

    for _, row in df[df["experiment"] == "reranking"].iterrows():
        i += 1
        enabled = row["variant"] != "off"
        records.append(
            ExperimentRecord(
                experiment_id=f"EXP-{i:03d}",
                date=DATE,
                description=f"Phase 12 reranking comparison: {row['variant']}",
                embedding_model=MINILM,
                chunk_size=512,
                chunk_overlap=64,
                retrieval_top_k=5,
                reranking_enabled=enabled,
                final_context_size=5,
                retrieval_metrics=metrics(row),
                notes=(
                    "Fixed: chunk_size=512, MiniLM, top_k=5 final. "
                    + ("Reranks top-20 down to top-5." if enabled else "No reranking.")
                ),
            )
        )

    return records


def _generation_records(start_index: int) -> list[ExperimentRecord]:
    df = pd.read_csv("data/evaluation/generation_eval_results.csv")
    answerable = df[df["answerable"]]
    no_evidence = df[~df["answerable"]]

    # Confirmed by Phase 13's log output: q001-q028 judged by qwen/qwen3.8-27b
    # (JSON-compliant); q029-q038 by allam-2-7b (fallback, JSON-noncompliant).
    reliable_ids = {f"q{n:03d}" for n in range(1, 29)}
    fallback_ids = {f"q{n:03d}" for n in range(29, 39)}

    records = []

    reliable = answerable[answerable["id"].isin(reliable_ids)]
    records.append(
        ExperimentRecord(
            experiment_id=f"EXP-{start_index:03d}",
            date=DATE,
            description="Phase 13 generation evaluation, questions q001-q028",
            embedding_model=MINILM,
            chunk_size=256,
            chunk_overlap=32,
            retrieval_top_k=5,
            reranking_enabled=True,
            final_context_size=5,
            generation_model="qwen/qwen3.8-27b",
            generation_metrics={
                "faithfulness": reliable["faithfulness"].mean(),
                "answer_relevance": reliable["answer_relevance"].mean(),
                "context_relevance": reliable["context_relevance"].mean(),
                "invalid_citations_total": float(reliable["invalid_citations"].sum()),
                "refusal_correctness": reliable["refusal_correct"].mean(),
            },
            notes=(
                f"n={len(reliable)}. Judge model followed the requested JSON output format "
                "reliably - these metrics are trustworthy."
            ),
        )
    )

    fallback_answerable = answerable[answerable["id"].isin(fallback_ids)]
    records.append(
        ExperimentRecord(
            experiment_id=f"EXP-{start_index + 1:03d}",
            date=DATE,
            description="Phase 13 generation evaluation, questions q029-q038 (answerable subset)",
            embedding_model=MINILM,
            chunk_size=256,
            chunk_overlap=32,
            retrieval_top_k=5,
            reranking_enabled=True,
            final_context_size=5,
            generation_model="allam-2-7b",
            generation_metrics={
                "faithfulness": fallback_answerable["faithfulness"].mean(),
                "answer_relevance": fallback_answerable["answer_relevance"].mean(),
                "context_relevance": fallback_answerable["context_relevance"].mean(),
                "invalid_citations_total": float(fallback_answerable["invalid_citations"].sum()),
                "refusal_correctness": fallback_answerable["refusal_correct"].mean(),
            },
            notes=(
                f"n={len(fallback_answerable)}. Used only after every higher-quality model's "
                "free-tier daily quota was exhausted. Did not reliably return the requested "
                "JSON format for judge prompts, so faithfulness here is an unreliable lower "
                "bound (see README) - kept separate from EXP for this reason, not averaged in."
            ),
        )
    )

    records.append(
        ExperimentRecord(
            experiment_id=f"EXP-{start_index + 2:03d}",
            date=DATE,
            description="Phase 13 refusal behavior on no-evidence questions (q031-q034)",
            embedding_model=MINILM,
            chunk_size=256,
            chunk_overlap=32,
            retrieval_top_k=5,
            reranking_enabled=True,
            final_context_size=5,
            generation_model="allam-2-7b",
            generation_metrics={
                "refusal_correctness_raw_heuristic": no_evidence["refusal_correct"].mean(),
                "invalid_citations_total": float(no_evidence["invalid_citations"].sum()),
            },
            notes=(
                "Raw automated score was 0% but manual inspection found 2/4 answers "
                "substantively declined using phrasing the refusal heuristic (since fixed) "
                "originally missed; one answer fabricated a citation to a real, in-range "
                "source number, which the Phase 10 citation checker cannot catch (it only "
                "validates that a cited number exists, not that the attributed content is "
                "real). See README for the full manual-inspection writeup."
            ),
        )
    )

    return records


def main() -> None:
    retrieval_records = _retrieval_records()
    generation_records = _generation_records(start_index=len(retrieval_records) + 1)
    all_records = retrieval_records + generation_records

    for record in all_records:
        log_experiment(record, EXPERIMENTS_PATH)

    logger.info("Logged %d experiment records to %s", len(all_records), EXPERIMENTS_PATH)


if __name__ == "__main__":
    main()
