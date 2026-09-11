from pathlib import Path

from cardiorag.evaluation.experiment_tracking import (
    experiments_to_dataframe,
    load_experiments,
    log_experiment,
)
from cardiorag.models import ExperimentRecord


def _make_record(**overrides) -> ExperimentRecord:
    defaults = dict(
        experiment_id="EXP-000",
        date="2026-09-10",
        description="a test experiment",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=256,
        chunk_overlap=32,
        retrieval_top_k=5,
        reranking_enabled=False,
        final_context_size=5,
        retrieval_metrics={"hit_rate": 0.8, "mrr": 0.6},
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_log_experiment_creates_file_and_appends(tmp_path: Path):
    path = tmp_path / "experiments.jsonl"
    log_experiment(_make_record(experiment_id="EXP-001"), path)
    log_experiment(_make_record(experiment_id="EXP-002"), path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_load_experiments_roundtrip(tmp_path: Path):
    path = tmp_path / "experiments.jsonl"
    record = _make_record(experiment_id="EXP-001", retrieval_metrics={"hit_rate": 0.85})
    log_experiment(record, path)

    loaded = load_experiments(path)

    assert len(loaded) == 1
    assert loaded[0] == record


def test_load_experiments_returns_empty_list_when_file_missing(tmp_path: Path):
    assert load_experiments(tmp_path / "does_not_exist.jsonl") == []


def test_experiments_to_dataframe_prefixes_metric_columns():
    records = [
        _make_record(
            experiment_id="EXP-001",
            retrieval_metrics={"hit_rate": 0.8},
            generation_metrics={"faithfulness": 0.9},
        )
    ]

    df = experiments_to_dataframe(records)

    assert df.loc[0, "retrieval_hit_rate"] == 0.8
    assert df.loc[0, "generation_faithfulness"] == 0.9
    assert df.loc[0, "experiment_id"] == "EXP-001"


def test_experiments_to_dataframe_handles_multiple_records():
    records = [_make_record(experiment_id="EXP-001"), _make_record(experiment_id="EXP-002")]

    df = experiments_to_dataframe(records)

    assert len(df) == 2
    assert list(df["experiment_id"]) == ["EXP-001", "EXP-002"]
