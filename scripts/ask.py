"""End-to-end demo of the full pipeline built so far:

    question -> retrieve -> rerank -> grounded generation -> cited answer

Requires OPENAI_API_KEY set in .env (or the environment) - load_provider_from_settings
raises a clear error immediately if it's missing, rather than failing deep
inside an HTTP call.
"""

import logging
import sys
from pathlib import Path

from cardiorag.config import settings
from cardiorag.generation.generator import generate_answer
from cardiorag.generation.providers import load_provider_from_settings
from cardiorag.retrieval.reranker import Reranker, retrieve_and_rerank
from cardiorag.retrieval.retriever import load_retriever

logging.basicConfig(level=logging.WARNING)

DEFAULT_QUESTION = "How does artificial intelligence improve cardiac MRI segmentation?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    provider = load_provider_from_settings(settings)  # raises early if misconfigured

    retriever = load_retriever(Path(settings.index_dir), settings.embedding_model, device="cpu")
    reranker = Reranker(settings.reranker_model, device="cpu")

    rerank_result = retrieve_and_rerank(retriever, reranker, question, retrieve_k=20, final_k=5)
    result = generate_answer(provider, question, rerank_result.reranked)

    print(f"Question: {result.question}\n")
    print(f"Answer:\n{result.answer}\n")
    print(f"Generated in {result.generation_seconds:.2f}s using {len(result.sources)} source(s):")
    for i, source in enumerate(result.sources, start=1):
        chunk = source.chunk
        print(f"  [Source {i}] {chunk.title!r} page(s)={chunk.page_numbers} score={source.score:.3f}")


if __name__ == "__main__":
    main()
