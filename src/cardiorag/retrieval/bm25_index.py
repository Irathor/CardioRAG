"""BM25 lexical retrieval and Reciprocal Rank Fusion (Phase 20 fix #5).

Dense embeddings capture semantic similarity well, but can dilute an exact,
rare term - an abbreviation, a specific dataset name, a numeric value -
that a lexical match would catch directly. BM25 is a classic term-frequency
sparse retrieval algorithm: not a neural model, and complementary to one
for exactly that reason.

Fusion uses reciprocal rank (Reciprocal Rank Fusion), not a weighted sum of
raw scores: cosine similarity (bounded [-1, 1]) and BM25's score (unbounded,
corpus-dependent) are not on comparable scales, so combining them by rank
avoids inventing an arbitrary weighting between two incomparable numbers.
"""

import re

from rank_bm25 import BM25Okapi

from cardiorag.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(self.chunks))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[Chunk, float]], k: int = 10
) -> list[tuple[Chunk, float]]:
    """Combine any number of ranked (chunk, score) lists into one ranking:
    score(chunk) = sum, over every list containing it, of 1/(k + rank in
    that list). Only rank matters, not the incoming scores - which is what
    makes this safe to use across lists with incomparable score scales.

    k=10, NOT the k=60 commonly cited from the original RRF paper (Cormack
    et al., 2009) - that value was tuned for TREC web-scale search over
    thousands of candidates, and measurably fails at this corpus's scale.
    Verified on a real query ("What is MOCOnet used for?", a rare CNN name
    named in exactly one paper): BM25 alone correctly ranked the matching
    chunk #1; with k=60 fused against a candidate_k=50 dense/BM25 pool, it
    disappeared from the fused top-5 entirely, because virtually any chunk
    appearing in both lists (even at mediocre rank) outscores a #1-in-one-
    list chunk once k exceeds the candidate pool size. With k=10, the same
    chunk correctly surfaces at rank 2. Lesson applied consistently with
    the rest of this project: a "standard" default is a starting point to
    measure, not a value to trust unchecked at a different scale.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunk_by_id[chunk.chunk_id] = chunk
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [(chunk_by_id[cid], fused_scores[cid]) for cid in ranked_ids]
