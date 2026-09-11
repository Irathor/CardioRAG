"""Reproducible text cleaning applied on top of raw PDF extraction.

Design principle: every function here is conservative. When a transformation
is ambiguous (most notably hyphenation at a line break), we choose the
option that cannot fabricate incorrect scientific terminology, even if it
leaves a little more surface noise than an aggressive cleaner would.

`clean_document` never mutates its input and never discards `raw_text` -
`cleaned_text` is stored alongside it on each page so both remain
inspectable.
"""

import re

from cardiorag.ingestion.text_analysis import find_repeated_first_lines, find_repeated_last_lines
from cardiorag.models import Document

# Joins a hyphen-broken line by removing only the newline, keeping the hyphen.
# "informa-\ntion" -> "informa-tion" (imperfect but not fabricated), while a
# genuine compound term like "T1-\nweighted" correctly becomes "T1-weighted".
# Blindly deleting the hyphen too would turn that into the wrong word
# "T1weighted" - unacceptable for a scientific corpus where such compounds
# are common domain terminology, not typographic accidents.
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\n(?=[a-zA-Z])")

# Soft/discretionary hyphen (U+00AD): a PDF-generator-inserted, invisible
# marker for an optional line-break point. Unlike an ASCII "-" it is never
# genuine punctuation, so there is no ambiguity to preserve - when it precedes
# the newline where the break actually happened, both characters are simply
# deleted (no hyphen, no space): "measure\xad\nments" -> "measurements".
_SOFT_HYPHEN_LINEBREAK_RE = re.compile("\xad\n")

_RUN_OF_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")

# Copyright/license boilerplate phrases, taken from real front-matter text
# observed across this project's own corpus (not guessed) - e.g. "This is an
# open-access article distributed under the terms of the Creative Commons
# Attribution License (CC BY)... No use, distribution or reproduction is
# permitted which does not comply with these terms." A sentence containing
# any of these is journal/publisher boilerplate, never a scientific claim.
_BOILERPLATE_PHRASES = (
    "creative commons",
    "cc by",
    "open access article",
    "open-access article",
    "no use, distribution or reproduction",
    "accepted academic practice",
    "unrestricted use, distribution, and reproduction",
    "all rights reserved",
    "provided the original author",
)
# Simple sentence-boundary heuristic (period/!/? followed by a capitalized
# word) - not a full sentence tokenizer. An occasional mis-split only
# changes how much surrounding text is removed alongside a real match, not
# whether the check itself is correct.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def strip_soft_hyphens(text: str) -> str:
    """Remove soft hyphens (U+00AD), joining any word they split with nothing
    in between. Any stray soft hyphen not at a line break is deleted too,
    since it is never meant to be visible content."""
    text = _SOFT_HYPHEN_LINEBREAK_RE.sub("", text)
    return text.replace("\xad", "")


def dehyphenate(text: str) -> str:
    """Remove line breaks that split a hyphenated token, keeping the hyphen."""
    return _HYPHEN_LINEBREAK_RE.sub("-", text)


def unwrap_soft_line_breaks(text: str) -> str:
    """Join lines within a blank-line-delimited block into one line.

    PDF text extraction emits one line per visual line of a column, which
    rarely matches sentence or paragraph boundaries. We treat a blank line
    as the only reliable paragraph boundary and join everything else -
    simple, and correct for the common case of prose-heavy scientific text.

    Known limitation: a list or table without blank lines between items
    will be merged into a single line too. Fixing this properly needs
    layout-aware extraction (e.g. PyMuPDF's block/dict mode), which is out
    of scope for this minimal version.
    """
    blocks = re.split(r"\n\s*\n", text)
    joined_blocks = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if lines:
            joined_blocks.append(" ".join(lines))
    return "\n\n".join(joined_blocks)


def strip_boilerplate_sentences(text: str) -> str:
    """Remove sentences matching known copyright/licensing boilerplate
    phrases - front-matter noise distinct from the references-section
    problem (already excluded earlier in the pipeline). Real example still
    visible in retrieval results as of Phase 12/19 testing: "This is an
    open-access article distributed under the terms of the Creative Commons
    Attribution License (CC BY)... No use, distribution or reproduction is
    permitted which does not comply with these terms."

    Deliberately narrow: this catches the copyright/license *sentence*
    specifically. Author-affiliation lists and editorial-workflow metadata
    (RECEIVED/REVIEWED/CITATION blocks) are a different, harder problem -
    they have no equally reliable structural marker (unlike "References",
    a standalone "Abstract" heading was checked and found present in only
    5 of 8 real corpus papers, too unreliable a signal to build on) and are
    NOT addressed by this function. See the README's Limitations section.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in sentences if not any(phrase in s.lower() for phrase in _BOILERPLATE_PHRASES)]
    return " ".join(kept)


def collapse_whitespace(text: str) -> str:
    """Normalize runs of spaces/tabs to one space and cap blank-line runs at one."""
    text = _RUN_OF_SPACES_RE.sub(" ", text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text.strip()


def _strip_matched_line(text: str, repeated_lines: set[str], *, position: str) -> str:
    """Blank out the first or last non-empty line of `text` if it matches a
    line already confirmed (by the caller) to repeat across pages. Blanking
    rather than deleting keeps every other line's position stable."""
    if not repeated_lines:
        return text
    lines = text.split("\n")
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return text
    target_index = non_empty_indices[0] if position == "first" else non_empty_indices[-1]
    if lines[target_index].strip() in repeated_lines:
        lines[target_index] = ""
    return "\n".join(lines)


def clean_page_text(text: str, *, header_lines: set[str], footer_lines: set[str]) -> str:
    """Run the full per-page cleaning pipeline in order: strip confirmed
    running headers/footers first (they're matched against raw lines), then
    dehyphenate (needs the original newlines), then unwrap soft line breaks,
    then strip copyright/license boilerplate sentences (needs prose-shaped
    text to split into sentences), then collapse residual whitespace."""
    text = _strip_matched_line(text, header_lines, position="first")
    text = _strip_matched_line(text, footer_lines, position="last")
    text = strip_soft_hyphens(text)
    text = dehyphenate(text)
    text = unwrap_soft_line_breaks(text)
    text = strip_boilerplate_sentences(text)
    text = collapse_whitespace(text)
    return text


def clean_document(document: Document) -> Document:
    """Return a new Document with `cleaned_text` populated on every page.
    `document` itself is not modified."""
    raw_pages = [page.raw_text for page in document.pages]
    header_lines = set(find_repeated_first_lines(raw_pages))
    footer_lines = set(find_repeated_last_lines(raw_pages))

    cleaned_pages = [
        page.model_copy(
            update={
                "cleaned_text": clean_page_text(
                    page.raw_text, header_lines=header_lines, footer_lines=footer_lines
                )
            }
        )
        for page in document.pages
    ]

    return document.model_copy(update={"pages": cleaned_pages})
