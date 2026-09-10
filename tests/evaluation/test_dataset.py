from pathlib import Path

import pytest

from cardiorag.evaluation.dataset import load_evaluation_dataset
from cardiorag.ingestion.pdf_loader import load_corpus
from cardiorag.models import EvaluationCategory

DATASET_PATH = Path("data/evaluation/eval_dataset.jsonl")


@pytest.fixture(scope="module")
def dataset():
    return load_evaluation_dataset(DATASET_PATH)


def test_dataset_file_exists():
    assert DATASET_PATH.exists()


def test_dataset_size_is_within_the_requested_range(dataset):
    assert 30 <= len(dataset) <= 50


def test_dataset_ids_are_unique(dataset):
    ids = [example.id for example in dataset]
    assert len(ids) == len(set(ids))


def test_every_category_is_represented(dataset):
    categories_present = {example.category for example in dataset}
    assert categories_present == set(EvaluationCategory)


def test_answerable_examples_have_expected_documents(dataset):
    for example in dataset:
        if example.answerable:
            assert example.expected_document_ids, f"{example.id} is answerable but cites no document"


def test_no_evidence_examples_are_unanswerable_with_no_expected_documents(dataset):
    no_evidence = [e for e in dataset if e.category == EvaluationCategory.NO_EVIDENCE]
    assert len(no_evidence) >= 1
    for example in no_evidence:
        assert example.answerable is False
        assert example.expected_document_ids == []


def test_expected_pages_are_positive(dataset):
    for example in dataset:
        for page in example.expected_pages:
            assert page > 0, f"{example.id} has a non-positive expected page {page}"


def test_expected_document_ids_match_the_real_corpus(dataset):
    """Guards against a stale/typo'd document_id in the hand-authored dataset:
    every id it references must actually exist in the current corpus."""
    result = load_corpus(Path("data/corpus"))
    real_document_ids = {doc.document_id for doc in result.documents}

    for example in dataset:
        for doc_id in example.expected_document_ids:
            assert doc_id in real_document_ids, (
                f"{example.id} references document_id {doc_id!r} which is not in the "
                "current corpus - the PDF may have changed or the id was mistyped"
            )
