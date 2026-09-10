# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 8-9 — grounded generation + LLM provider abstraction. `src/cardiorag/generation/`
builds the evidence context (`[Source N]` blocks with title/page/DOI kept bound to their text),
enforces a strict system prompt (answer only from sources, refuse explicitly when evidence is
insufficient, never invent citations, flag disagreement between sources, non-diagnostic
disclaimer), and calls an injected `LLMProvider` (Phase 9's abstraction) to generate the answer.
Only `OpenAIProvider` is implemented so far; `huggingface_local`/`ollama` raise
`NotImplementedError` rather than fake support. If zero chunks were retrieved, the system refuses
before ever calling the LLM. `scripts/ask.py` runs the full pipeline end-to-end
(retrieve -> rerank -> generate) — **not yet tested against a real LLM call**: it requires
`OPENAI_API_KEY`, which isn't configured in this environment. Verified instead that it fails
fast with a clear error when the key is missing, rather than failing deep inside an HTTP call.

Phase 7 (reranking) added a cross-encoder second stage, with an honest experimental finding worth
keeping in mind for the sources this pipeline surfaces: reranking did **not** cleanly fix the
bibliography/citation-list problem noted in Phase 5/6 on the real corpus's messier chunk
boundaries — added complexity didn't obviously improve results there. See the limitations list
for remediation options (most likely fix is upstream, in chunking).

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
