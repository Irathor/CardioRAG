"""Phase 3 chunking experiment: compare chunk_size/overlap configurations
against the real corpus and report descriptive statistics.

This does NOT measure retrieval quality (Recall@K, MRR, etc.) - there is no
embedding index yet. It measures what chunking itself produces: how many
chunks, how full each one is, how much text a page boundary forces to split.
Which configuration retrieves best is decided empirically in Phase 12, once
we can actually query an index; this script only gives us the descriptive
groundwork to reason about that choice later.
"""

import logging
from pathlib import Path

import pandas as pd

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus

logging.basicConfig(level=logging.WARNING)

CONFIGS = [
    ChunkingConfig(chunk_size=256, chunk_overlap=32),
    ChunkingConfig(chunk_size=512, chunk_overlap=64),
    ChunkingConfig(chunk_size=768, chunk_overlap=96),
]


def main() -> None:
    result = load_corpus(Path("data/corpus"))
    if result.failures:
        raise RuntimeError(f"Ingestion failures, aborting: {result.failures}")

    cleaned_documents = [clean_document(doc) for doc in result.documents]

    rows = []
    for config in CONFIGS:
        all_chunks = []
        for document in cleaned_documents:
            all_chunks.extend(chunk_document(document, config))

        token_counts = pd.Series([c.token_count for c in all_chunks])
        rows.append(
            {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "num_chunks": len(all_chunks),
                "avg_tokens_per_chunk": round(token_counts.mean(), 1),
                "min_tokens": int(token_counts.min()),
                "max_tokens": int(token_counts.max()),
                "chunks_under_50pct_full": int((token_counts < config.chunk_size * 0.5).sum()),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
