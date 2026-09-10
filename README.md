# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 11 — evaluation dataset. `data/evaluation/eval_dataset.jsonl` holds 38
hand-authored, manually-verified questions (17 factual, 7 comparison, 3 synthesis, 3 multi-paper,
4 no-evidence, 4 misleading) built by actually reading the corpus's cleaned text and spot-checking
facts against the extracted page content — not fabricated from general knowledge. Each example
carries expected document id(s)/page(s), a reference answer, and an answerability label.
`tests/evaluation/test_dataset.py` validates structure (id uniqueness, category coverage,
consistency between `answerable` and expected sources) and cross-checks every referenced
document id against the *live* corpus, so a changed PDF would be caught rather than silently
going stale. This produces ground truth for Phase 12 (retrieval metrics) — no Recall@K/MRR is
computed yet.

Phase 10 (citations) added `src/cardiorag/generation/citations.py`, which adds two
things that don't depend on the LLM: (1) mechanical verification — extracting every `[Source N]`
marker from generated text and flagging any N outside the range of sources actually provided, so
"never invent citations" is checked, not just requested in the prompt; (2) `build_citation_list()`,
which groups the flat `RetrievedChunk` list by document (a paper can contribute several chunks)
into presentation-ready `Citation` objects — built strictly from retrieval metadata, never from
the LLM's own text. Verified against the real index: the 5 reranked chunks for a sample query
collapsed into 2 distinct documents, and a deliberately fabricated `[Source 99]` reference was
correctly flagged as invalid. The UI to actually let a user inspect this (Phase 16) doesn't exist
yet.

Underneath, Phases 8-9 built grounded generation (`src/cardiorag/generation/`): evidence context
blocks (`[Source N]` with title/page/DOI kept bound to their text), a strict system prompt
(evidence-only claims, explicit refusal when insufficient, no invented citations, flag source
disagreement, non-diagnostic disclaimer), and a swappable `LLMProvider` (Phase 9) — only
`OpenAIProvider` is implemented so far. `scripts/ask.py` wires the full pipeline
(retrieve -> rerank -> generate) but **hasn't been tested against a real LLM call yet**: it
requires `OPENAI_API_KEY`, not configured in this environment. Verified instead that it fails
fast with a clear error when the key is missing.

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
