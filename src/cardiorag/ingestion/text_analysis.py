"""Cross-page text analysis shared by extraction-issue detection (pdf_loader)
and cleaning (cleaner): finding lines repeated across many pages, which is
how running headers/footers are identified either way.
"""

# A first/last line repeated on at least this many pages is treated as a
# running header/footer rather than a coincidence.
HEADER_FOOTER_MIN_REPEATS = 3


def _line_occurrences(pages: list[str], position: int) -> dict[str, list[int]]:
    occurrences: dict[str, list[int]] = {}
    for i, text in enumerate(pages):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            occurrences.setdefault(lines[position], []).append(i + 1)
    return occurrences


def find_repeated_first_lines(
    pages: list[str], min_repeats: int = HEADER_FOOTER_MIN_REPEATS
) -> dict[str, list[int]]:
    """Lines that open a page and recur on at least `min_repeats` pages -> candidate headers."""
    return {
        line: page_numbers
        for line, page_numbers in _line_occurrences(pages, 0).items()
        if len(page_numbers) >= min_repeats
    }


def find_repeated_last_lines(
    pages: list[str], min_repeats: int = HEADER_FOOTER_MIN_REPEATS
) -> dict[str, list[int]]:
    """Lines that close a page and recur on at least `min_repeats` pages -> candidate footers."""
    return {
        line: page_numbers
        for line, page_numbers in _line_occurrences(pages, -1).items()
        if len(page_numbers) >= min_repeats
    }
