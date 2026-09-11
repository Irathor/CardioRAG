import pytest

from cardiorag.evaluation.retrieval_metrics import (
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_true_when_any_relevant():
    assert hit_rate([False, False, True]) == 1.0


def test_hit_rate_zero_when_none_relevant():
    assert hit_rate([False, False]) == 0.0


def test_hit_rate_zero_for_empty_list():
    assert hit_rate([]) == 0.0


def test_precision_at_k_computes_fraction_relevant():
    assert precision_at_k([True, False, True, False]) == 0.5


def test_precision_at_k_zero_for_empty_list():
    assert precision_at_k([]) == 0.0


def test_recall_at_k_computes_fraction_of_total_relevant_found():
    assert recall_at_k([True, False], num_relevant_total=4) == 0.25


def test_recall_at_k_is_vacuously_one_when_nothing_relevant_exists():
    assert recall_at_k([False, False], num_relevant_total=0) == 1.0


def test_reciprocal_rank_of_first_position():
    assert reciprocal_rank([True, False, False]) == 1.0


def test_reciprocal_rank_of_second_position():
    assert reciprocal_rank([False, True, False]) == 0.5


def test_reciprocal_rank_zero_when_none_found():
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k_matches_hand_computed_value():
    # relevant at positions 1 and 3 (1-indexed), 2 relevant items exist total.
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG (best case: relevant items at positions 1,2) = 1/log2(2) + 1/log2(3) ~= 1.63093
    result = ndcg_at_k([True, False, True], num_relevant_total=2)
    assert result == pytest.approx(1.5 / 1.63093, abs=1e-4)


def test_ndcg_at_k_is_one_for_perfect_ranking():
    result = ndcg_at_k([True, True, False], num_relevant_total=2)
    assert result == pytest.approx(1.0)


def test_ndcg_at_k_zero_when_nothing_relevant_exists():
    assert ndcg_at_k([False, False], num_relevant_total=0) == 0.0
