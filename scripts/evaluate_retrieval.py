"""Phase 12: measure retrieval quality against the ground-truth dataset
(Phase 11) instead of guessing. Compares, one variable at a time:

    - chunk size (256 / 512 / 768 tokens, fixed embedding model + top_k)
    - embedding model (general-purpose vs. scientific-paper-specific)
    - top_k (3 / 5 / 10 / 20)
    - reranking on vs. off

Changing one variable at a time (rather than a full factorial grid) is a
deliberate scope tradeoff: it isolates each variable's marginal effect
cheaply, at the cost of not exploring interactions between them (e.g.
"does reranking help more at a particular chunk size?"). That's an honest
limitation of this study, not an oversight.
"""

import logging
from pathlib import Path

import pandas as pd

from cardiorag.chunking.chunker import ChunkingConfig, chunk_document
from cardiorag.embeddings.embedder import Embedder
from cardiorag.evaluation.dataset import load_evaluation_dataset
from cardiorag.evaluation.evaluator import evaluate_retrieval
from cardiorag.ingestion.cleaner import clean_document
from cardiorag.ingestion.pdf_loader import load_corpus
from cardiorag.models import Chunk, RetrievedChunk
from cardiorag.retrieval.reranker import Reranker
from cardiorag.retrieval.vector_store import VectorStore, build_vector_store

logging.basicConfig(level=logging.WARNING)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCIENTIFIC_MODEL = "sentence-transformers/allenai-specter"
DEFAULT_K = 5


def _build_store(chunks: list[Chunk], model_name: str) -> tuple[VectorStore, Embedder]:
    embedder = Embedder(model_name, device="cpu")
    vectors = embedder.embed([c.text for c in chunks]).vectors
    return build_vector_store(vectors, chunks), embedder


def _make_retrieve_fn(store: VectorStore, embedder: Embedder):
    def retrieve_fn(question: str, k: int) -> list[RetrievedChunk]:
        query_vector = embedder.embed([question]).vectors[0]
        results = store.search(query_vector, top_k=k)
        return [RetrievedChunk(chunk=c, score=s) for c, s in results]

    return retrieve_fn


def _make_rerank_retrieve_fn(store: VectorStore, embedder: Embedder, reranker: Reranker, retrieve_k: int):
    def retrieve_fn(question: str, k: int) -> list[RetrievedChunk]:
        query_vector = embedder.embed([question]).vectors[0]
        candidates = [
            RetrievedChunk(chunk=c, score=s) for c, s in store.search(query_vector, top_k=retrieve_k)
        ]
        return reranker.rerank(question, candidates, top_k=k).reranked

    return retrieve_fn


def main() -> None:
    examples = load_evaluation_dataset(Path("data/evaluation/eval_dataset.jsonl"))
    result = load_corpus(Path("data/corpus"))
    cleaned_documents = [clean_document(doc) for doc in result.documents]

    rows = []

    # --- 1. Chunk size sweep (fixed model=MiniLM, top_k=5, no reranking) ---
    chunks_by_size: dict[int, list[Chunk]] = {}
    for chunk_size in [256, 512, 768]:
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_size // 8)
        chunks = [c for doc in cleaned_documents for c in chunk_document(doc, config)]
        chunks_by_size[chunk_size] = chunks

        store, embedder = _build_store(chunks, DEFAULT_MODEL)
        metrics = evaluate_retrieval(
            examples, chunks, _make_retrieve_fn(store, embedder), k=DEFAULT_K
        )
        rows.append(
            {"experiment": "chunk_size", "variant": str(chunk_size), **vars(metrics)}
        )

    # --- 2. Embedding model comparison (fixed chunk_size=512, top_k=5, no reranking) ---
    default_chunks = chunks_by_size[DEFAULT_CHUNK_SIZE]
    minilm_store, minilm_embedder = _build_store(default_chunks, DEFAULT_MODEL)
    minilm_metrics = evaluate_retrieval(
        examples, default_chunks, _make_retrieve_fn(minilm_store, minilm_embedder), k=DEFAULT_K
    )
    rows.append({"experiment": "embedding_model", "variant": "MiniLM (general)", **vars(minilm_metrics)})

    specter_store, specter_embedder = _build_store(default_chunks, SCIENTIFIC_MODEL)
    specter_metrics = evaluate_retrieval(
        examples, default_chunks, _make_retrieve_fn(specter_store, specter_embedder), k=DEFAULT_K
    )
    rows.append({"experiment": "embedding_model", "variant": "SPECTER (scientific)", **vars(specter_metrics)})

    # --- 3. top_k sweep (fixed chunk_size=512, MiniLM, no reranking) ---
    for k in [3, 5, 10, 20]:
        metrics = evaluate_retrieval(
            examples, default_chunks, _make_retrieve_fn(minilm_store, minilm_embedder), k=k
        )
        rows.append({"experiment": "top_k", "variant": f"k={k}", **vars(metrics)})

    # --- 4. Reranking on vs. off (fixed chunk_size=512, MiniLM, final k=5) ---
    no_rerank_metrics = evaluate_retrieval(
        examples, default_chunks, _make_retrieve_fn(minilm_store, minilm_embedder), k=DEFAULT_K
    )
    rows.append({"experiment": "reranking", "variant": "off", **vars(no_rerank_metrics)})

    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    rerank_retrieve_fn = _make_rerank_retrieve_fn(minilm_store, minilm_embedder, reranker, retrieve_k=20)
    rerank_metrics = evaluate_retrieval(examples, default_chunks, rerank_retrieve_fn, k=DEFAULT_K)
    rows.append({"experiment": "reranking", "variant": "on (retrieve 20 -> rerank 5)", **vars(rerank_metrics)})

    df = pd.DataFrame(rows)
    df = df[["experiment", "variant", "hit_rate", "mrr", "precision_at_k", "recall_at_k", "ndcg_at_k", "num_questions"]]
    print(df.to_string(index=False))
    df.to_csv("data/evaluation/retrieval_eval_results.csv", index=False)
    print("\nSaved to data/evaluation/retrieval_eval_results.csv")


if __name__ == "__main__":
    main()
