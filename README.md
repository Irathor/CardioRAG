# CardioRAG

A production-style Retrieval-Augmented Generation (RAG) system specialized in scientific
literature on cardiovascular magnetic resonance (CMR), cardiovascular imaging, and AI applied
to cardiovascular medicine.

**Status:** Phase 12 — retrieval evaluation. `src/cardiorag/evaluation/retrieval_metrics.py`
(Hit Rate, Precision@K, Recall@K, MRR, nDCG) and `evaluator.py` (relevance judged by
document+page overlap, not exact chunk_id, since chunk boundaries shift across chunk_size
configs) measure retrieval against the Phase 11 ground truth. `scripts/evaluate_retrieval.py`
ran the full comparison against all 34 answerable questions and saved results to
`data/evaluation/retrieval_eval_results.csv`:

| experiment | variant | hit_rate | mrr | precision@k | recall@k | ndcg@k |
|---|---|---:|---:|---:|---:|---:|
| chunk_size | 256 | 0.82 | 0.61 | 0.24 | 0.18 | 0.30 |
| chunk_size | 512 | 0.71 | 0.45 | 0.19 | 0.23 | 0.26 |
| chunk_size | 768 | 0.68 | 0.47 | 0.17 | 0.28 | 0.28 |
| embedding_model | MiniLM (general) | 0.71 | 0.45 | 0.19 | 0.23 | 0.26 |
| embedding_model | SPECTER (scientific) | 0.44 | 0.30 | 0.12 | 0.14 | 0.16 |
| top_k | k=3/5/10/20 | 0.53/0.71/0.82/0.91 | 0.41/0.45/0.47/0.48 | ↓ as k grows | ↑ as k grows | ↑ as k grows |
| reranking | off | 0.71 | 0.45 | 0.19 | 0.23 | 0.26 |
| reranking | on (20→5) | 0.85 | 0.71 | 0.29 | 0.36 | 0.43 |

**Two findings that overturn earlier assumptions, with data instead of intuition:**
1. **SPECTER (scientific-paper-specific) retrieves *worse* than general-purpose MiniLM on every
   metric.** Plausible reason: SPECTER is trained for document-level (title+abstract) similarity
   via citation graphs, not fine-grained passage retrieval for question-answering - the actual
   task here. "Domain-specific" is not automatically "better" without measuring against the
   actual retrieval task.
2. **Reranking helps substantially on average** (MRR 0.45 -> 0.71, nDCG 0.26 -> 0.43), reversing
   the qualitative, single-question impression from Phase 7 that it "didn't obviously help." One
   example misled; 34 measured questions did not.

Chunk size shows a real precision/recall tradeoff (256 tokens: best Hit Rate/MRR, worst Recall;
768: the reverse). Given the project's emphasis on citation trustworthiness over exhaustive
coverage, the default is now **256 tokens** (`ChunkingConfig` in `chunker.py`) - the persisted
embeddings/index were rebuilt accordingly (724 chunks, up from 364).

**Update - the bibliography problem is now fixed at the source.** A spot-check at 256 tokens
initially showed the reference-list ranking problem (Phase 5-7) getting *more* pronounced, not
less. The actual fix: `pdf_loader.py` detects a standalone "References"/"Bibliography" heading
line in each PDF's RAW text (before cleaning merges it into the citation list, destroying the
signal) - verified against all 8 real corpus papers with zero false positives, despite each using
a different citation style ("[1]", "1.", "01.", or unnumbered author-year). Every page from that
heading onward is flagged `is_references_section=True`, and `chunker.py` now skips those pages
entirely rather than post-filtering chunks after the fact. Result: **395 chunks instead of
724** (45% of the previous chunk set was pure bibliography), and re-running Phase 12's evaluation
confirms a real, measured improvement at the default config, not just a qualitative impression:

| metric | before fix | after fix |
|---|---:|---:|
| hit_rate | 0.82 | 0.85 |
| mrr | 0.61 | 0.63 |
| precision@k | 0.24 | 0.25 |
| recall@k | 0.18 | 0.19 |
| ndcg@k | 0.30 | 0.31 |

**Honest remaining gap**: a qualitative check after the fix still surfaces non-substantive text
of a *different* kind - author-affiliation lists and copyright/licensing boilerplate ("publication
in this journal is cited, in accordance with accepted academic practice..."). The references fix
solved exactly what it targeted; front-matter boilerplate is a related but distinct problem,
still open.

Underneath, Phase 11 — evaluation dataset. `data/evaluation/eval_dataset.jsonl` holds 38
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
