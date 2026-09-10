"""Citation verification and formatting.

Two distinct concerns live here:

1. Verification: the system prompt tells the LLM to reference sources as
   "[Source N]" and never invent one. This module checks that promise
   mechanically - it does not trust the model's compliance, it checks it -
   by extracting every [Source N] marker from the generated text and
   flagging any N outside the range of sources actually provided.

2. Formatting: turning the flat list of RetrievedChunk sources fed to the
   LLM into a presentation-ready citation list, grouped by document (a
   single paper can contribute more than one chunk). This is built
   strictly from retrieval metadata, never from the LLM's own text, so a
   citation can never claim a title/DOI/page that wasn't actually
   retrieved.
"""

import re

from cardiorag.models import Citation, RetrievedChunk

_SOURCE_MARKER_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def extract_cited_source_numbers(answer_text: str) -> set[int]:
    """Every distinct source number the LLM referenced as [Source N]."""
    return {int(match) for match in _SOURCE_MARKER_RE.findall(answer_text)}


def find_invalid_citations(answer_text: str, num_sources: int) -> set[int]:
    """Source numbers the LLM cited that don't correspond to any source it
    was actually given (N < 1 or N > num_sources) - a fabricated citation,
    caught mechanically rather than trusted."""
    cited = extract_cited_source_numbers(answer_text)
    return {n for n in cited if n < 1 or n > num_sources}


def find_uncited_sources(answer_text: str, num_sources: int) -> set[int]:
    """Source numbers that were provided as evidence but never referenced
    in the answer - not necessarily a problem (the model may have used
    some sources for context without citing all of them), but a useful
    signal for later evaluation of citation/faithfulness behavior (Phase 13)."""
    cited = extract_cited_source_numbers(answer_text)
    all_sources = set(range(1, num_sources + 1))
    return all_sources - cited


def build_citation_list(sources: list[RetrievedChunk]) -> list[Citation]:
    """Group retrieved chunks by document into presentation-ready citations,
    preserving first-appearance order. Pages and chunk_ids are merged
    across every chunk from the same document that contributed."""
    citations_by_doc: dict[str, Citation] = {}
    order: list[str] = []

    for retrieved in sources:
        chunk = retrieved.chunk
        if chunk.document_id not in citations_by_doc:
            citations_by_doc[chunk.document_id] = Citation(
                document_id=chunk.document_id,
                title=chunk.title,
                doi=chunk.doi,
                source_filename=chunk.source_filename,
                pages=list(chunk.page_numbers),
                chunk_ids=[chunk.chunk_id],
                max_score=retrieved.score,
            )
            order.append(chunk.document_id)
        else:
            existing = citations_by_doc[chunk.document_id]
            merged_pages = sorted(set(existing.pages) | set(chunk.page_numbers))
            existing.pages[:] = merged_pages
            existing.chunk_ids.append(chunk.chunk_id)
            existing.max_score = max(existing.max_score, retrieved.score)

    return [citations_by_doc[doc_id] for doc_id in order]
