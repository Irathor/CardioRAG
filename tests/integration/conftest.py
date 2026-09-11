"""Shared fixtures for integration tests: a small synthetic corpus (two
PDFs with distinct topics and a References section each) exercised through
the real ingestion/cleaning/chunking pipeline, so tests don't depend on the
actual contents of data/corpus/.
"""

from pathlib import Path

import pymupdf
import pytest


def _write_pdf(path: Path, pages: list[str], metadata: dict[str, str] | None = None) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        # insert_text draws from a single point with no line wrapping - a
        # paragraph wider than the page gets silently clipped mid-word.
        # insert_textbox wraps within a rectangle, which is what these
        # longer, realistic integration-test paragraphs actually need.
        rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, text, fontsize=11)
    if metadata:
        doc.set_metadata(metadata)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    _write_pdf(
        corpus_dir / "cardiac.pdf",
        pages=[
            "Deep learning enables automated segmentation of the left ventricle in "
            "cardiac magnetic resonance imaging, reducing analysis time for clinicians.",
            "Convolutional neural networks trained on large annotated datasets achieve "
            "high accuracy in delineating myocardial borders on cine MRI sequences.",
            "References",
            "[1] Smith J, Doe A. Deep learning for cardiac segmentation. J Cardiovasc Imaging 2021.",
        ],
        metadata={"title": "Deep Learning for Cardiac MRI Segmentation"},
    )
    _write_pdf(
        corpus_dir / "unrelated.pdf",
        pages=[
            "Quarterly earnings reports show significant growth in the technology sector "
            "driven by increased consumer spending on electronics.",
            "References",
            "[1] Jones B. Market trends in consumer electronics. Econ Rev 2020.",
        ],
        metadata={"title": "Technology Sector Earnings Report"},
    )
    return corpus_dir
