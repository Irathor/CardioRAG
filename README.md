# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 7 — reranking. `src/cardiorag/retrieval/reranker.py` adds a cross-encoder
second stage (`cross-encoder/ms-marco-MiniLM-L-6-v2`): unlike the bi-encoder used for dense
retrieval, a cross-encoder scores `[query, passage]` jointly through the transformer's attention
instead of comparing independently-computed vectors, at the cost of not being precomputable —
it only runs on the small candidate set dense retrieval already narrowed down (e.g. top-20 -> top-5).

**Honest experimental finding** (`scripts/compare_reranking.py`, run against the real index):
reranking did **not** cleanly fix the bibliography/citation-list problem noted in Phase 5/6 — a
targeted unit test with two clearly-contrasted texts showed the cross-encoder *can* prefer
substantive explanation over a citation, but on the real corpus's actual (messier) chunk
boundaries, citation-adjacent text still ranked highly after reranking. This is the Phase 7
warning made concrete: added complexity (a second model, ~0.9s extra latency for 20 candidates)
did not obviously improve results here. The more likely fix is upstream, in chunking
(filtering/flagging reference sections) — see the limitations list. Quantitative confirmation
either way needs Phase 12's ground-truth evaluation. Generation is not implemented yet.

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
