"""Phase 4: embed the full corpus with the configured embedding model and
persist the result to data/processed/, so the corpus never needs to be
re-embedded just to start the app or run an experiment again.
"""

import logging
from pathlib import Path

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.config import settings
from cardiorag.embeddings.embedder import Embedder
from cardiorag.embeddings.store import save_embeddings
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    result = load_corpus(settings.corpus_dir)
    if result.failures:
        logger.warning("Ingestion failures (continuing with the rest): %s", result.failures)

    cleaned_documents = [clean_document(doc) for doc in result.documents]
    chunks = [
        chunk for doc in cleaned_documents for chunk in chunk_document(doc, ChunkingConfig())
    ]
    logger.info("Chunked %d documents into %d chunks", len(cleaned_documents), len(chunks))

    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    embedding_result = embedder.embed([c.text for c in chunks], show_progress_bar=True)

    save_embeddings(settings.processed_dir, chunks, embedding_result)
    logger.info(
        "Persisted %d embeddings (model=%s, dim=%d) to %s",
        len(chunks),
        embedding_result.model_name,
        embedding_result.dimension,
        settings.processed_dir,
    )


if __name__ == "__main__":
    main()
