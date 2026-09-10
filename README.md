# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 5 — vector index. Embeddings (Phase 4) are indexed with FAISS
(`src/cardiorag/retrieval/vector_store.py`) using exact inner-product search (`IndexFlatIP`):
since every embedding is L2-normalized, inner product *is* cosine similarity, so no
cosine-specific index is needed. The index and its aligned chunk metadata persist together to
`indexes/` (`scripts/build_index.py`) and are always loaded/saved as a pair, so they can never
silently drift out of sync. Turning a text question into a query vector, and everything after
that, is Phase 6 (retrieval) — this phase only does vector-to-vector search and metadata lookup.

Known limitation surfaced during manual verification: nearest-neighbor results can include
bibliography/reference-list text (it shares vocabulary with the query but isn't a substantive
claim) — worth revisiting once reranking (Phase 7) and evaluation (Phase 12) exist. Generation
is not implemented yet.

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
