"""Phase 4 experiment: compare a general-purpose embedding model against a
model built specifically for scientific literature (SPECTER, trained on
paper citation graphs) on the real corpus.

This records exactly what Phase 4 asks for: model, embedding dimension,
normalization, and encode time. It is NOT a retrieval-quality comparison -
that requires an index and ground-truth relevant chunks, which don't exist
until Phase 5/12. Whichever model "wins" here is only whichever is faster;
which one retrieves better is a separate, later question.
"""

import logging
from pathlib import Path

import pandas as pd

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.embeddings.embedder import Embedder
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus

logging.basicConfig(level=logging.WARNING)

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # general-purpose, 384-dim
    "sentence-transformers/allenai-specter",  # scientific-paper-specific, 768-dim
]


def main() -> pd.DataFrame:
    result = load_corpus(Path("data/corpus"))
    if result.failures:
        raise RuntimeError(f"Ingestion failures, aborting: {result.failures}")

    cleaned_documents = [clean_document(doc) for doc in result.documents]
    chunks = [
        chunk for doc in cleaned_documents for chunk in chunk_document(doc, ChunkingConfig())
    ]
    texts = [chunk.text for chunk in chunks]
    print(f"Embedding {len(texts)} chunks from {len(cleaned_documents)} documents\n")

    rows = []
    for model_name in MODELS:
        embedder = Embedder(model_name, device="auto")
        embedding_result = embedder.embed(texts, show_progress_bar=True)
        rows.append(
            {
                "model": model_name,
                "dimension": embedding_result.dimension,
                "normalized": embedding_result.normalized,
                "num_chunks": len(texts),
                "encode_seconds": round(embedding_result.encode_seconds, 2),
                "chunks_per_second": round(
                    len(texts) / embedding_result.encode_seconds, 1
                ),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
