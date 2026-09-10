"""Phase 6 experiment: observe how retrieved results change across
different top_k values against the real index.

This does NOT measure retrieval quality (Recall@K, Precision@K, etc.) -
that requires a ground-truth evaluation dataset of (question, relevant
chunk) pairs, which is Phase 11. It only shows what varying top_k actually
returns and how similarity scores decay, so the tradeoff (more context vs.
more noise/tokens) is visible before Phase 12 measures it properly.
"""

import logging
from pathlib import Path

import pandas as pd

from cardiorag.config import settings
from cardiorag.retrieval.retriever import load_retriever

logging.basicConfig(level=logging.WARNING)

TOP_K_VALUES = [3, 5, 10, 20]

SAMPLE_QUESTIONS = [
    "How does artificial intelligence improve cardiac MRI segmentation?",
    "What are the challenges of using retrieval-augmented generation in healthcare?",
]


def main() -> pd.DataFrame:
    retriever = load_retriever(Path(settings.index_dir), settings.embedding_model, device="cpu")

    rows = []
    for question in SAMPLE_QUESTIONS:
        for k in TOP_K_VALUES:
            results = retriever.retrieve(question, top_k=k)
            scores = [r.score for r in results]
            rows.append(
                {
                    "question": question[:40] + "...",
                    "top_k": k,
                    "num_results": len(results),
                    "best_score": round(max(scores), 4) if scores else None,
                    "worst_score": round(min(scores), 4) if scores else None,
                    "unique_documents": len({r.chunk.document_id for r in results}),
                }
            )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
