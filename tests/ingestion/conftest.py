"""Shared helpers for building small, deterministic synthetic PDFs in tests
so we never depend on the content of the real corpus in data/corpus/."""

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Returns a factory: make_pdf(name, pages=["text for page 1", ...], metadata={...}) -> Path"""

    def _make(
        name: str,
        pages: list[str],
        metadata: dict[str, str] | None = None,
    ) -> Path:
        doc = pymupdf.open()
        for page_text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), page_text, fontsize=11)
        if metadata:
            doc.set_metadata(metadata)
        path = tmp_path / name
        doc.save(str(path))
        doc.close()
        return path

    return _make
