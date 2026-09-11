from pathlib import Path

import pytest

from cardiorag.evaluation.experiment_tracking import load_experiments

EXPERIMENTS_PATH = Path("data/evaluation/experiments.jsonl")


@pytest.fixture(scope="module")
def experiments():
    return load_experiments(EXPERIMENTS_PATH)


def test_experiments_log_exists():
    assert EXPERIMENTS_PATH.exists()


def test_experiment_ids_are_unique(experiments):
    ids = [e.experiment_id for e in experiments]
    assert len(ids) == len(set(ids))


def test_every_experiment_has_either_retrieval_or_generation_metrics(experiments):
    for record in experiments:
        assert record.retrieval_metrics or record.generation_metrics, (
            f"{record.experiment_id} logs no metrics at all"
        )


def test_generation_experiments_record_which_llm_was_used(experiments):
    for record in experiments:
        if record.generation_metrics:
            assert record.generation_model, (
                f"{record.experiment_id} has generation metrics but no generation_model recorded"
            )
