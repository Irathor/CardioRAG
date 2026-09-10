"""Standard information-retrieval metrics, computed from a per-query ranked
list of binary relevance flags.

Every function takes `relevant: list[bool]`, already in ranked order and
already truncated to whatever K was retrieved - the flag at position i says
whether the i-th retrieved item counts as relevant to the question.
`num_relevant_total` (where required) is the count of relevant items that
exist anywhere in the index, not just what was retrieved - Recall and nDCG
need to know what was missed, which the ranked list alone can't tell you.
"""

import math


def hit_rate(relevant: list[bool]) -> float:
    """1.0 if at least one relevant item appears anywhere in the ranked list, else 0.0."""
    return 1.0 if any(relevant) else 0.0


def precision_at_k(relevant: list[bool]) -> float:
    """Fraction of the retrieved list that is actually relevant."""
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)


def recall_at_k(relevant: list[bool], num_relevant_total: int) -> float:
    """Fraction of all relevant items in the index that appear in this
    ranked list. Vacuously 1.0 when there is nothing relevant to find."""
    if num_relevant_total == 0:
        return 1.0
    return sum(relevant) / num_relevant_total


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / rank of the first relevant item (1-indexed), or 0.0 if none found."""
    for i, is_rel in enumerate(relevant, start=1):
        if is_rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: list[bool], num_relevant_total: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance: relevant
    items earlier in the ranking contribute more, normalized against the
    best possible ordering of the same number of relevant items.

    Undefined (returns 0.0) when num_relevant_total is 0 - callers should
    exclude such queries from an aggregate average rather than average in
    a meaningless 0.
    """
    dcg = sum(
        (1.0 if is_rel else 0.0) / math.log2(i + 1) for i, is_rel in enumerate(relevant, start=1)
    )
    ideal_hits = min(num_relevant_total, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
