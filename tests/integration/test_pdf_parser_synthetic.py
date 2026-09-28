import pytest

from research_agent.rag.pdf_parser import AcademicPDFLoader


@pytest.mark.integration
def test_synthetic_academic_pdf_preserves_page_trace_and_column_order(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("PDF_MIN_TEXT_CHARS", "0")
    monkeypatch.setenv("PDF_EXTRACT_TABLES", "false")
    path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    for page_number in range(3):
        page = pdf.new_page(width=600, height=800)
        page.insert_text((70, 28), "Research Conference 2026", fontsize=8)
        page.insert_text((295, 785), str(page_number + 1), fontsize=8)
        page.insert_text((60, 100), "1. Introduction", fontsize=15)
        page.insert_textbox(
            fitz.Rect(60, 140, 280, 350),
            "Left column explains the retrieval method. " * 8,
            fontsize=10,
        )
        page.insert_textbox(
            fitz.Rect(320, 140, 540, 350),
            "Right column reports evidence and limitations. " * 8,
            fontsize=10,
        )
    pdf.save(path)
    pdf.close()

    documents = AcademicPDFLoader(path).load()
    assert len(documents) == 3
    assert documents[0].metadata["page_start"] == 1
    assert documents[0].metadata["parser"].startswith("academic_pdf_pymupdf")
    assert "Research Conference" not in documents[0].page_content
    assert documents[0].page_content.index("Left column") < documents[0].page_content.index(
        "Right column"
    )


@pytest.mark.integration
def test_mixed_pdf_propagates_blank_scan_page_warning(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("PDF_MIN_TEXT_CHARS", "30")
    monkeypatch.setenv("PDF_OCR_MODE", "disabled")
    monkeypatch.setenv("PDF_EXTRACT_TABLES", "false")
    path = tmp_path / "mixed-paper.pdf"
    pdf = fitz.open()
    text_page = pdf.new_page(width=600, height=800)
    text_page.insert_textbox(
        fitz.Rect(60, 100, 540, 300),
        "This digital page contains enough searchable academic text. " * 4,
        fontsize=10,
    )
    pdf.new_page(width=600, height=800)  # Simulate an image-only scan page.
    pdf.save(path)
    pdf.close()

    documents = AcademicPDFLoader(path).load()
    assert len(documents) == 1
    assert "page_2:low_text_density_possible_scan" in documents[0].metadata[
        "parser_warnings"
    ]

