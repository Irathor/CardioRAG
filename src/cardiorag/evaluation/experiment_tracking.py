"""Structured experiment tracking (Phase 14).

Every significant retrieval/generation experiment becomes one
ExperimentRecord, appended to a JSONL log - so comparing experiments means
reading one consistent, typed record format instead of remembering which
ad hoc script produced which CSV under which unwritten-down configuration
(exactly the trap Phase 12/13's scripts started falling into: their
"fixed" baseline parameters lived only as constants inside the script).
"""

import json
from pathlib import Path

import pandas as pd

from cardiorag.models import ExperimentRecord


def log_experiment(record: ExperimentRecord, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json())
        f.write("\n")


def load_experiments(path: Path) -> list[ExperimentRecord]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ExperimentRecord.model_validate(json.loads(line)) for line in f if line.strip()]


def experiments_to_dataframe(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Flatten records into one comparison table, prefixing metric dict keys
    so retrieval and generation metrics never collide (e.g. two different
    "hit_rate"-shaped things could theoretically coexist)."""
    rows = []
    for r in records:
        row = {
            "experiment_id": r.experiment_id,
            "date": r.date,
            "description": r.description,
            "embedding_model": r.embedding_model,
            "generation_model": r.generation_model,
            "chunk_size": r.chunk_size,
            "chunk_overlap": r.chunk_overlap,
            "retrieval_top_k": r.retrieval_top_k,
            "reranking_enabled": r.reranking_enabled,
            "final_context_size": r.final_context_size,
            "latency_seconds": r.latency_seconds,
        }
        row.update({f"retrieval_{k}": v for k, v in r.retrieval_metrics.items()})
        row.update({f"generation_{k}": v for k, v in r.generation_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)
