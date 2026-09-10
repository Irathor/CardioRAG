# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 6 — retrieval. `src/cardiorag/retrieval/retriever.py` closes the query loop:
question text -> embed with the same model the index was built with -> FAISS search -> ranked
`RetrievedChunk` results (chunk + similarity score). `Retriever` validates that the embedder's
output dimension matches the index's at construction time, so a model/index mismatch fails
immediately instead of returning meaningless scores. `top_k` is a plain parameter of `retrieve()`;
`scripts/compare_top_k.py` shows how score decay and source diversity change across
top_k in {3, 5, 10, 20} on the real index — not a quality measurement (Recall@K etc. needs the
ground-truth dataset from Phase 11), just what varying it actually returns.

Known limitations (see the full list with remediation options requested separately): nearest-neighbor
results can surface bibliography/reference-list text; embedding scores (0.5-0.7 cosine similarity)
suggest a general-purpose model may be under-discriminating on specialized medical vocabulary.
Reranking and generation are not implemented yet.

CardioRAG is a research/educational project. It is **not** a medical diagnostic system and its
output must never be treated as medical advice.

## Project layout

```
app/                    Streamlit UI (Phase 16)
data/
  corpus/               Source PDFs (git-ignored — large/copyrighted binaries)
  references/           Reference/gold documents for evaluation
  processed/            Cleaned text, chunks (generated, git-ignored)
  evaluation/           Hand-authored evaluation dataset (versioned)
indexes/                Persisted FAISS indexes (generated, git-ignored)
notebooks/experiments/  Exploratory notebooks (chunking/embedding comparisons, etc.)
src/cardiorag/
  ingestion/            PDF loading, metadata extraction (Phase 1)
  chunking/             Token-aware chunking (Phase 3)
  embeddings/           Embedding model abstraction (Phase 4)
  retrieval/            Vector store, retriever, reranker (Phases 5-7)
  generation/           Context building, prompts, LLM generation (Phases 8-9)
  evaluation/           Retrieval & generation metrics (Phases 12-13)
  api/                  FastAPI backend (Phase 15)
  config.py             Typed settings loaded from environment variables
  models.py             Shared Pydantic domain models (added in Phase 1)
tests/                  pytest suite
scripts/                CLI entry points (build_index.py, evaluate.py — later phases)
```

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"       # core + test tooling only, for now
cp .env.example .env          # then fill in secrets as needed
pytest
```

Additional dependency groups (`ingestion`, `ml`, `api`, `ui`, `eval`) are installed
incrementally as each corresponding phase is implemented — there's no reason to pull in
PyTorch/FAISS before we're embedding anything.

## Roadmap

See the milestone plan: PDF ingestion → chunking → embeddings → vector search → retrieval →
grounded generation → citations → reranking → evaluation → API + UI + Docker. Each milestone
is developed and tested incrementally.
