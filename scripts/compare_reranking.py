"""Phase 7 experiment: qualitative before/after comparison of dense-only
retrieval vs. dense retrieval + cross-encoder reranking, on the real index.

This is NOT a quantitative retrieval-quality measurement (no Recall@K, no
nDCG) - that needs the ground-truth relevant-chunk labels from Phase 11,
which don't exist yet. What we CAN honestly show now is whether reranking
changes the ranking in the direction we'd want: does it push citation/
bibliography-style text down and substantive explanatory text up? That's
a concrete, real question this script answers with actual output, not a
fabricated metric.
"""

import logging
from pathlib import Path

from cardiorag.config import settings
from cardiorag.retrieval.reranker import Reranker, retrieve_and_rerank
from cardiorag.retrieval.retriever import load_retriever

logging.basicConfig(level=logging.WARNING)

QUESTION = "How does artificial intelligence improve cardiac MRI segmentation?"
RETRIEVE_K = 20
FINAL_K = 5


def _print_results(label: str, results) -> None:
    print(f"\n--- {label} ---")
    for rank, r in enumerate(results, start=1):
        preview = r.chunk.text[:110].replace("\n", " ")
        print(f"[{rank}] score={r.score:.4f} page(s)={r.chunk.page_numbers} text={preview!r}")


def main() -> None:
    retriever = load_retriever(Path(settings.index_dir), settings.embedding_model, device="cpu")
    reranker = Reranker(settings.reranker_model, device="cpu")

    dense_only = retriever.retrieve(QUESTION, top_k=FINAL_K)
    _print_results(f"Dense retrieval only (top {FINAL_K})", dense_only)

    rerank_result = retrieve_and_rerank(
        retriever, reranker, QUESTION, retrieve_k=RETRIEVE_K, final_k=FINAL_K
    )
    _print_results(
        f"Dense retrieval (top {RETRIEVE_K}) + cross-encoder rerank (top {FINAL_K})",
        rerank_result.reranked,
    )
    print(f"\nRerank time for {RETRIEVE_K} candidates: {rerank_result.rerank_seconds:.3f}s")


if __name__ == "__main__":
    main()
