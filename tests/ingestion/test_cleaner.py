from cardiorag.ingestion.cleaner import (
    clean_document,
    clean_page_text,
    collapse_whitespace,
    dehyphenate,
    strip_boilerplate_sentences,
    strip_soft_hyphens,
    unwrap_soft_line_breaks,
)
from cardiorag.models import Document, DocumentMetadata, PageContent


def test_dehyphenate_removes_linebreak_but_keeps_hyphen():
    assert dehyphenate("cardio-\nvascular magnetic resonance") == "cardio-vascular magnetic resonance"


def test_dehyphenate_preserves_genuine_compound_terms():
    # This is the whole point of the conservative design: a real compound term
    # like "T1-weighted" must not become "T1weighted".
    assert dehyphenate("T1-\nweighted images") == "T1-weighted images"


def test_dehyphenate_ignores_line_breaks_without_hyphen():
    assert dehyphenate("Hello\nWorld") == "Hello\nWorld"


def test_strip_soft_hyphens_joins_word_broken_at_soft_hyphen():
    # This is the real artifact PyMuPDF surfaces for justified scientific PDFs:
    # a soft hyphen (U+00AD) at the exact point a word was broken for layout.
    assert strip_soft_hyphens("measure\xad\nments of function.") == "measurements of function."


def test_strip_soft_hyphens_removes_stray_occurrence_without_linebreak():
    assert strip_soft_hyphens("soft\xadhyphen") == "softhyphen"


def test_clean_page_text_fully_reconstructs_soft_hyphenated_word():
    text = "Cine imaging provides accurate and reproducible measure\xad\nments of function."
    cleaned = clean_page_text(text, header_lines=set(), footer_lines=set())

    assert "measurements" in cleaned
    assert "\xad" not in cleaned


def test_unwrap_soft_line_breaks_joins_wrapped_paragraph():
    text = "This is a sentence\nthat was wrapped\nacross three lines."
    assert unwrap_soft_line_breaks(text) == "This is a sentence that was wrapped across three lines."


def test_unwrap_soft_line_breaks_preserves_paragraph_boundary():
    text = "First paragraph\nwrapped here.\n\nSecond paragraph\nalso wrapped."
    result = unwrap_soft_line_breaks(text)
    assert result == "First paragraph wrapped here.\n\nSecond paragraph also wrapped."


def test_collapse_whitespace_normalizes_runs_of_spaces():
    assert collapse_whitespace("a   b\tc") == "a b c"


def test_collapse_whitespace_caps_blank_line_runs():
    text = "a\n\n\n\n\nb"
    assert collapse_whitespace(text) == "a\n\nb"


def test_collapse_whitespace_strips_leading_and_trailing_whitespace():
    assert collapse_whitespace("  \n hello \n  ") == "hello"


def test_strip_boilerplate_sentences_removes_real_cc_by_license_text():
    # Real text observed in this project's own corpus (3.pdf).
    text = (
        "Deep learning improves myocardial characterisation in cardiac MRI. "
        "This is an open-access article distributed under the terms of the Creative Commons "
        "Attribution License (CC BY). No use, distribution or reproduction is permitted which "
        "does not comply with these terms. "
        "The following sections review recent advances in this field."
    )

    cleaned = strip_boilerplate_sentences(text)

    assert "Deep learning improves myocardial characterisation" in cleaned
    assert "The following sections review recent advances" in cleaned
    assert "Creative Commons" not in cleaned
    assert "No use, distribution or reproduction" not in cleaned


def test_strip_boilerplate_sentences_removes_real_plos_license_text():
    # Real text observed in this project's own corpus (5.pdf).
    text = (
        "Retrieval augmented generation improves healthcare LLMs. "
        "This is an open access article distributed under the terms of the Creative Commons "
        "Attribution License, which permits unrestricted use, distribution, and reproduction in "
        "any medium, provided the original author and source are credited. "
        "We conducted a systematic literature review."
    )

    cleaned = strip_boilerplate_sentences(text)

    assert "Retrieval augmented generation improves healthcare LLMs" in cleaned
    assert "We conducted a systematic literature review" in cleaned
    assert "unrestricted use" not in cleaned


def test_strip_boilerplate_sentences_keeps_ordinary_text_untouched():
    text = "Cardiac MRI provides high spatial resolution. It is used to assess ventricular function."

    assert strip_boilerplate_sentences(text) == text


def test_clean_page_text_preserves_numerical_values_units_and_abbreviations():
    text = (
        "LVEF was 55.3% (95% CI: 52.1-58.5) at 1.5T using T1 mapping (ms), "
        "consistent with CMR findings."
    )
    cleaned = clean_page_text(text, header_lines=set(), footer_lines=set())

    for fragment in ["55.3%", "95% CI", "52.1-58.5", "1.5T", "T1 mapping (ms)", "CMR"]:
        assert fragment in cleaned


def test_clean_page_text_strips_only_confirmed_repeated_header():
    text = "Journal of Cardiovascular Imaging\nActual page content here."
    cleaned = clean_page_text(
        text, header_lines={"Journal of Cardiovascular Imaging"}, footer_lines=set()
    )

    assert "Journal of Cardiovascular Imaging" not in cleaned
    assert "Actual page content here." in cleaned


def test_clean_page_text_does_not_strip_unconfirmed_lines():
    text = "A line that looks like a header.\nBody content."
    cleaned = clean_page_text(text, header_lines={"Some other header"}, footer_lines=set())

    assert "A line that looks like a header." in cleaned


def _make_document(pages_raw_text: list[str]) -> Document:
    pages = [
        PageContent(
            page_number=i + 1,
            raw_text=text,
            char_count=len(text),
            is_empty=len(text.strip()) == 0,
        )
        for i, text in enumerate(pages_raw_text)
    ]
    metadata = DocumentMetadata(filename="synthetic.pdf", num_pages=len(pages))
    return Document(document_id="test-doc", metadata=metadata, pages=pages)


def test_clean_document_populates_cleaned_text_without_touching_raw_text():
    document = _make_document(["cardio-\nvascular   disease   study."])

    cleaned_document = clean_document(document)

    original_page = cleaned_document.pages[0]
    assert original_page.raw_text == "cardio-\nvascular   disease   study."
    assert original_page.cleaned_text == "cardio-vascular disease study."


def test_clean_document_does_not_mutate_input():
    document = _make_document(["some text here"])

    clean_document(document)

    assert document.pages[0].cleaned_text is None


def test_clean_document_removes_headers_repeated_across_pages():
    header = "Circulation: Cardiovascular Imaging"
    document = _make_document([f"{header}\nContent {i}." for i in range(4)])

    cleaned_document = clean_document(document)

    for page in cleaned_document.pages:
        assert header not in page.cleaned_text
        assert page.raw_text.startswith(header)  # raw text is untouched
