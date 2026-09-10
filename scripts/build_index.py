"""Phase 5: build a FAISS vector index from the embeddings persisted in
Phase 4 (data/processed/) and persist it to indexes/, so the index doesn't
need to be rebuilt every time the app starts - only when the corpus or
embedding model changes.
"""

import logging
from pathlib import Path

from cardiorag.config import settings
from cardiorag.embeddings.store import load_embeddings
from cardiorag.retrieval.vector_store import build_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    vectors, chunks = load_embeddings(settings.processed_dir, settings.embedding_model)
    logger.info("Loaded %d embeddings (dim=%d) from %s", len(chunks), vectors.shape[1], settings.processed_dir)

    store = build_vector_store(vectors, chunks)
    store.save(settings.index_dir)
    logger.info("Persisted FAISS index (%d vectors) to %s", store.index.ntotal, settings.index_dir)


if __name__ == "__main__":
    main()
