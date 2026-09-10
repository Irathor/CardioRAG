from cardiorag.ingestion.pdf_loader import load_pdf


def test_uses_embedded_title_when_present(make_pdf):
    path = make_pdf(
        "titled.pdf",
        pages=["Some body text about T1 mapping."],
        metadata={"title": "Native T1 Mapping in Hypertrophic Cardiomyopathy"},
    )

    document = load_pdf(path)

    assert document.metadata.title == "Native T1 Mapping in Hypertrophic Cardiomyopathy"


def test_falls_back_to_first_line_when_embedded_title_is_junk(make_pdf):
    path = make_pdf(
        "untitled.pdf",
        pages=["Deep Learning for Cardiac MRI Segmentation: A Review"],
        metadata={"title": ""},
    )

    document = load_pdf(path)

    assert document.metadata.title == "Deep Learning for Cardiac MRI Segmentation: A Review"


def test_extracts_doi_from_page_text(make_pdf):
    path = make_pdf(
        "with_doi.pdf",
        pages=["Some title line here.\nhttps://doi.org/10.1016/j.jcmg.2021.01.002"],
    )

    document = load_pdf(path)

    assert document.metadata.doi == "10.1016/j.jcmg.2021.01.002"


def test_extracts_publication_year_from_page_text(make_pdf):
    path = make_pdf("with_year.pdf", pages=["Published in 2019 by the Society."])

    document = load_pdf(path)

    assert document.metadata.publication_year == 2019


def test_parses_semicolon_separated_authors(make_pdf):
    path = make_pdf(
        "authors.pdf",
        pages=["Body text."],
        metadata={"author": "Smith, John; Doe, Jane"},
    )

    document = load_pdf(path)

    assert document.metadata.authors == ["Smith, John", "Doe, Jane"]


def test_missing_fields_stay_none_rather_than_fabricated(make_pdf):
    path = make_pdf("sparse.pdf", pages=["x" * 20])

    document = load_pdf(path)

    assert document.metadata.doi is None
    assert document.metadata.publication_year is None
    assert document.metadata.authors == []
