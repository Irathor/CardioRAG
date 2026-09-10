"""Token-aware chunking of a cleaned Document into overlapping Chunks.

Chunking is treated as a tunable ML experiment, not a hardcoded constant:
chunk size and overlap directly shape retrieval quality. A chunk too large
dilutes the embedding with irrelevant text and buries the relevant sentence;
a chunk too small loses the surrounding context needed to interpret it, and
overlap trades a bit of redundancy for not losing evidence that straddles a
chunk boundary. `ChunkingConfig` makes both explicit so Phase 12 can compare
configurations empirically instead of guessing.
"""

from dataclasses import dataclass

from cardiorag.chunking.tokenization import get_tokenizer
from cardiorag.models import Chunk, Document


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 512  # tokens; 512 is also all-MiniLM-L6-v2's max sequence length
    chunk_overlap: int = 64  # tokens
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def _page_text(page) -> str:  # noqa: ANN001 - PageContent, kept loose to avoid a circular import
    return page.cleaned_text if page.cleaned_text is not None else page.raw_text


def _build_page_text_stream(document: Document) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate every page's text into one stream (preferring cleaned_text,
    falling back to raw_text for a page that hasn't been cleaned), and return
    the (char_start, char_end, page_number) ranges needed to attribute each
    resulting chunk back to the page(s) it was drawn from.
    """
    non_empty_pages = [page for page in document.pages if _page_text(page)]

    pieces: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for i, page in enumerate(non_empty_pages):
        text = _page_text(page)
        start = cursor
        pieces.append(text)
        cursor += len(text)
        page_ranges.append((start, cursor, page.page_number))
        if i < len(non_empty_pages) - 1:
            # Separator so tokens from adjacent pages never merge into one token.
            pieces.append("\n\n")
            cursor += 2
    return "".join(pieces), page_ranges


def _pages_for_span(page_ranges: list[tuple[int, int, int]], start: int, end: int) -> list[int]:
    pages = {
        page_number
        for range_start, range_end, page_number in page_ranges
        if range_start < end and range_end > start
    }
    return sorted(pages)


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split `document`'s cleaned text into overlapping, token-bounded chunks.

    Returns an empty list for a document with no usable text rather than
    raising - an all-image or all-empty-page PDF is a legitimate (if
    unhelpful) input.
    """
    config = config or ChunkingConfig()
    tokenizer = get_tokenizer(config.embedding_model)

    full_text, page_ranges = _build_page_text_stream(document)
    if not full_text.strip():
        return []

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    num_tokens = len(offsets)

    step = config.chunk_size - config.chunk_overlap
    chunks: list[Chunk] = []
    start_token = 0
    index = 0

    while start_token < num_tokens:
        end_token = min(start_token + config.chunk_size, num_tokens)
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        chunk_text = full_text[char_start:char_end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{index:04d}",
                    document_id=document.document_id,
                    text=chunk_text,
                    token_count=end_token - start_token,
                    page_numbers=_pages_for_span(page_ranges, char_start, char_end),
                    title=document.metadata.title,
                    doi=document.metadata.doi,
                    source_filename=document.metadata.filename,
                )
            )
            index += 1

        if end_token == num_tokens:
            break
        start_token += step

    return chunks
