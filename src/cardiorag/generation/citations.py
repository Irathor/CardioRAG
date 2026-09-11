"""Citation verification and formatting.

Three distinct concerns live here:

1. Range verification: the system prompt tells the LLM to reference sources
   as "[Source N]" and never invent one. This checks that promise
   mechanically - it does not trust the model's compliance, it checks it -
   by extracting every [Source N] marker and flagging any N outside the
   range of sources actually provided.

2. Content verification: range validity alone isn't enough. Manual
   inspection during Phase 13 found a real case the range check cannot
   catch - an answer attributed a verbatim quote to a real, in-range
   source number, but that quote does not appear anywhere in that source's
   actual text (a fabricated citation to a genuine source). This is a
   deterministic, LLM-free check by design, consistent with the rest of
   this module: it only catches VERBATIM quoted-and-attributed
   fabrications, not paraphrased ones - a broader, LLM-based faithfulness
   check exists separately in evaluation/generation_metrics.py for that.

3. Formatting: turning the flat list of RetrievedChunk sources fed to the
   LLM into a presentation-ready citation list, grouped by document. Built
   strictly from retrieval metadata, never from the LLM's own text, so a
   citation can never claim a title/DOI/page that wasn't actually
   retrieved.
"""

import difflib
import re

from cardiorag.models import Citation, RetrievedChunk

_SOURCE_MARKER_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
# Quoted spans of at least 15 characters - shorter quotes ("the" "MRI") are
# too generic to meaningfully check and would produce noisy false positives.
_QUOTE_RE = re.compile(r'["“]([^"”]{15,})["”]')
# How far to look for a [Source N] marker around a quote to consider it
# "attributed" to that source - covers both "[Source 1] states: '...'" and
# "'...' [Source 1]" phrasing styles.
_ATTRIBUTION_WINDOW_BEFORE = 100
_ATTRIBUTION_WINDOW_AFTER = 30
# Below this fraction of the quote's length matching contiguously in the
# source text, treat the quote as fabricated rather than paraphrased/reflowed.
_MIN_MATCH_RATIO = 0.6


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


def _find_attributed_source_number(answer_text: str, quote_start: int, quote_end: int) -> int | None:
    """Look for a [Source N] marker near a quote - before it first (the more
    common "[Source N] states: '...'" phrasing), then after ("'...' [Source N]")."""
    before = answer_text[max(0, quote_start - _ATTRIBUTION_WINDOW_BEFORE) : quote_start]
    before_matches = list(_SOURCE_MARKER_RE.finditer(before))
    if before_matches:
        return int(before_matches[-1].group(1))  # nearest to the quote = last match in the window

    after = answer_text[quote_end : quote_end + _ATTRIBUTION_WINDOW_AFTER]
    after_match = _SOURCE_MARKER_RE.search(after)
    return int(after_match.group(1)) if after_match else None


def find_fabricated_quotes(answer_text: str, sources: list[RetrievedChunk]) -> list[dict]:
    """Find quoted strings attributed to a specific [Source N] that do not
    actually appear (even approximately) in that source's retrieved text.

    Returns a list of {source_number, quoted_text, match_ratio} for each
    quote that failed to match. An out-of-range source_number is skipped
    here (find_invalid_citations already reports that separately).
    """
    findings = []
    for quote_match in _QUOTE_RE.finditer(answer_text):
        quote = quote_match.group(1)
        source_number = _find_attributed_source_number(
            answer_text, quote_match.start(), quote_match.end()
        )
        if source_number is None or not (1 <= source_number <= len(sources)):
            continue

        source_text = sources[source_number - 1].chunk.text
        if quote.lower() in source_text.lower():
            continue  # exact substring match - fast path, no fabrication

        longest_match = difflib.SequenceMatcher(
            None, quote.lower(), source_text.lower()
        ).find_longest_match(0, len(quote), 0, len(source_text))
        match_ratio = longest_match.size / len(quote)

        if match_ratio < _MIN_MATCH_RATIO:
            findings.append(
                {"source_number": source_number, "quoted_text": quote, "match_ratio": match_ratio}
            )

    return findings


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
