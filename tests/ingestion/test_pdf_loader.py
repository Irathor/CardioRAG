from pathlib import Path

from cardiorag.ingestion.pdf_loader import (
    compute_document_id,
    load_corpus,
    load_pdf,
)
from cardiorag.models import ExtractionIssueType


def test_compute_document_id_is_deterministic_and_content_addressed():
    id_a = compute_document_id(b"same content")
    id_b = compute_document_id(b"same content")
    id_c = compute_document_id(b"different content")

    assert id_a == id_b
    assert id_a != id_c


def test_load_pdf_extracts_pages_with_traceable_metadata(make_pdf):
    path = make_pdf("paper.pdf", pages=["Introduction to CMR.", "Methods section."])

    document = load_pdf(path)

    assert len(document.pages) == 2
    assert document.pages[0].page_number == 1
    assert "Introduction to CMR." in document.pages[0].raw_text
    assert document.pages[1].page_number == 2
    assert "Methods section." in document.pages[1].raw_text
    assert document.metadata.filename == "paper.pdf"
    assert document.metadata.num_pages == 2
    assert document.document_id == compute_document_id(path.read_bytes())


def test_load_pdf_flags_empty_page(make_pdf):
    path = make_pdf("with_blank.pdf", pages=["Real content on this page.", ""])

    document = load_pdf(path)

    assert document.pages[1].is_empty is True
    assert document.pages[0].is_empty is False
    empty_issues = [i for i in document.issues if i.issue_type == ExtractionIssueType.EMPTY_PAGE]
    assert len(empty_issues) == 1
    assert empty_issues[0].page_number == 2


def test_load_pdf_flags_hyphenation_artifacts(make_pdf):
    path = make_pdf("hyphenated.pdf", pages=["cardio-\nvascular magnetic resonance imaging"])

    document = load_pdf(path)

    hyphen_issues = [
        i for i in document.issues if i.issue_type == ExtractionIssueType.HYPHENATION_ARTIFACT
    ]
    assert len(hyphen_issues) == 1
    assert hyphen_issues[0].page_number == 1


def test_load_pdf_flags_excessive_whitespace(make_pdf):
    noisy_text = "Result" * 5 + (" " * 200)
    path = make_pdf("whitespace.pdf", pages=[noisy_text])

    document = load_pdf(path)

    ws_issues = [
        i for i in document.issues if i.issue_type == ExtractionIssueType.EXCESSIVE_WHITESPACE
    ]
    assert len(ws_issues) == 1


def test_load_pdf_flags_broken_line_wrapping(make_pdf):
    short_lines = "\n".join("word" for _ in range(15))
    path = make_pdf("short_lines.pdf", pages=[short_lines])

    document = load_pdf(path)

    wrap_issues = [
        i for i in document.issues if i.issue_type == ExtractionIssueType.BROKEN_LINE_WRAPPING
    ]
    assert len(wrap_issues) == 1


def test_load_pdf_flags_duplicated_headers_across_pages(make_pdf):
    header = "Journal of Cardiovascular Imaging - Volume 12"
    pages = [f"{header}\nPage body content number {i}." for i in range(4)]
    path = make_pdf("headers.pdf", pages=pages)

    document = load_pdf(path)

    header_issues = [
        i for i in document.issues if i.issue_type == ExtractionIssueType.DUPLICATED_HEADER
    ]
    assert len(header_issues) == 4
    assert {i.page_number for i in header_issues} == {1, 2, 3, 4}


def test_load_pdf_marks_references_section_and_every_following_page(make_pdf):
    path = make_pdf(
        "with_refs.pdf",
        pages=[
            "Introduction to the study.",
            "References",
            "[1] Smith J. Some citation. Journal 2020.",
        ],
    )

    document = load_pdf(path)

    assert document.pages[0].is_references_section is False
    assert document.pages[1].is_references_section is True
    assert document.pages[2].is_references_section is True


def test_load_pdf_without_references_heading_leaves_all_pages_unflagged(make_pdf):
    path = make_pdf("no_refs.pdf", pages=["Just body text.", "More body text."])

    document = load_pdf(path)

    assert all(not page.is_references_section for page in document.pages)


def test_load_pdf_references_heading_detection_is_case_insensitive(make_pdf):
    path = make_pdf("caps_refs.pdf", pages=["Body text.", "REFERENCES", "[1] Citation."])

    document = load_pdf(path)

    assert document.pages[1].is_references_section is True


def test_load_pdf_does_not_flag_references_mentioned_mid_sentence(make_pdf):
    path = make_pdf(
        "mention.pdf",
        pages=["See the references [12] and [13] for details on this method."],
    )

    document = load_pdf(path)

    assert document.pages[0].is_references_section is False


def test_load_pdf_raises_on_unreadable_file(tmp_path: Path):
    bogus = tmp_path / "not_a_pdf.pdf"
    bogus.write_bytes(b"this is definitely not a pdf file")

    try:
        load_pdf(bogus)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_pdf.pdf" in str(exc)


def test_load_corpus_reports_failures_without_aborting(tmp_path: Path, make_pdf):
    make_pdf("good.pdf", pages=["Valid content about cardiac MRI."])
    (tmp_path / "corrupt.pdf").write_bytes(b"garbage")

    # make_pdf writes into its own tmp_path fixture; point load_corpus at that same dir
    corpus_dir = tmp_path
    result = load_corpus(corpus_dir)

    assert len(result.documents) == 1
    assert result.documents[0].metadata.filename == "good.pdf"
    assert len(result.failures) == 1
    assert result.failures[0].filename == "corrupt.pdf"
