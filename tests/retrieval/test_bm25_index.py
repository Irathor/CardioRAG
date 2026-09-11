from cardiorag.models import Chunk
from cardiorag.retrieval.bm25_index import BM25Index, reciprocal_rank_fusion


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        token_count=len(text.split()),
        page_numbers=[1],
        title="A Test Paper",
        doi=None,
        source_filename="synthetic.pdf",
    )


# --- BM25Index ---


def test_bm25_index_finds_exact_term_match():
    chunks = [
        _make_chunk("c0", "The mitochondria is the powerhouse of the cell."),
        _make_chunk("c1", "LVEF measures left ventricular ejection fraction directly."),
        _make_chunk("c2", "Quarterly earnings rose sharply this year."),
    ]
    index = BM25Index(chunks)

    results = index.search("LVEF ejection fraction", top_k=1)

    assert results[0][0].chunk_id == "c1"


def test_bm25_index_returns_empty_for_empty_corpus():
    index = BM25Index([])

    assert index.search("anything", top_k=5) == []


def test_bm25_index_clamps_top_k_to_available_chunks():
    chunks = [_make_chunk("c0", "some text"), _make_chunk("c1", "other text")]
    index = BM25Index(chunks)

    assert len(index.search("text", top_k=10)) == 2


# --- reciprocal_rank_fusion ---


def test_rrf_promotes_a_chunk_ranked_high_in_both_lists():
    a = _make_chunk("a", "chunk a")
    b = _make_chunk("b", "chunk b")
    c = _make_chunk("c", "chunk c")

    dense = [(a, 0.9), (b, 0.8), (c, 0.7)]
    bm25 = [(a, 5.0), (c, 4.0), (b, 3.0)]

    fused = reciprocal_rank_fusion(dense, bm25)

    assert fused[0][0].chunk_id == "a"  # ranked #1 in both lists


def test_rrf_includes_a_chunk_present_in_only_one_list():
    a = _make_chunk("a", "chunk a")
    b = _make_chunk("b", "chunk b")

    dense = [(a, 0.9)]
    bm25 = [(b, 5.0)]

    fused = reciprocal_rank_fusion(dense, bm25)

    fused_ids = {chunk.chunk_id for chunk, _ in fused}
    assert fused_ids == {"a", "b"}


def test_rrf_matches_hand_computed_score():
    a = _make_chunk("a", "chunk a")
    dense = [(a, 0.9)]  # rank 1
    bm25 = [(a, 5.0)]  # rank 1

    fused = reciprocal_rank_fusion(dense, bm25, k=60)

    expected = 1 / 61 + 1 / 61
    assert fused[0][1] == expected


def test_rrf_empty_lists_return_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_default_k_does_not_bury_a_single_source_top_match():
    """Regression test for a real finding (Phase 20 fix #5): with the RRF
    paper's commonly-cited k=60, several chunks ranked merely OK (not #1)
    in BOTH lists can collectively outscore a chunk ranked #1 in only ONE
    list - exactly what happened on a real query ("What is MOCOnet used
    for?") where BM25 alone correctly ranked the answer #1, but it
    vanished from the fused top-5 under k=60 once ~20 generically-related
    candidates each ranked moderately in both lists. This pins the default
    (k=10) to keep behaving correctly, and demonstrates the k=60 failure
    mode it was chosen to avoid.
    """
    exact_match = _make_chunk("exact_match", "the one true answer")
    # 20 filler chunks each ranked moderately (15th-34th) in BOTH lists -
    # a stand-in for "generically related" candidates neither search
    # strongly prefers, the situation that actually occurred. 14 disposable
    # placeholders per list push the fillers down to that moderate rank
    # without being part of what this test is actually checking.
    fillers = [_make_chunk(f"filler{i}", f"filler text {i}") for i in range(20)]
    dummy_dense = [_make_chunk(f"dummy_dense{i}", "x") for i in range(14)]
    dummy_bm25 = [_make_chunk(f"dummy_bm25{i}", "x") for i in range(14)]

    dense = [(d, 0.0) for d in dummy_dense] + [(f, 0.0) for f in fillers]
    bm25 = [(exact_match, 99.0)] + [(d, 0.0) for d in dummy_bm25] + [(f, 0.0) for f in fillers]

    def rank_of(fused: list, chunk_id: str) -> int:
        return next(i for i, (chunk, _) in enumerate(fused) if chunk.chunk_id == chunk_id)

    def best_filler_rank(fused: list) -> int:
        return min(rank_of(fused, f.chunk_id) for f in fillers)

    fused_default = reciprocal_rank_fusion(dense, bm25)  # default k=10
    assert rank_of(fused_default, "exact_match") < best_filler_rank(fused_default)

    fused_k60 = reciprocal_rank_fusion(dense, bm25, k=60)
    assert rank_of(fused_k60, "exact_match") > best_filler_rank(fused_k60)  # documented failure mode
