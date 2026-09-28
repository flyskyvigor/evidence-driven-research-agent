from research_agent.rag.pdf_parser import (
    PDFBlock,
    classify_pdf_blocks,
    join_pdf_lines,
    markdown_table,
    order_page_blocks,
    repeated_margin_signatures,
)


def _block(page, bbox, text, size=10.0, bold=False):
    return PDFBlock(page, bbox, text, font_size=size, bold=bold)


def test_two_column_reading_order_is_left_column_then_right_column():
    blocks = [
        _block(0, (320, 220, 540, 250), "right second"),
        _block(0, (60, 120, 280, 150), "left first"),
        _block(0, (320, 120, 540, 150), "right first"),
        _block(0, (60, 220, 280, 250), "left second"),
    ]
    ordered = order_page_blocks(blocks, page_width=600, page_height=800)
    assert [block.text for block in ordered] == [
        "left first", "left second", "right first", "right second"
    ]


def test_repeated_headers_and_page_numbers_are_detected_across_pages():
    pages = []
    for page in range(3):
        pages.append((600.0, 800.0, [
            _block(page, (60, 20, 300, 35), "Research Conference 2026"),
            _block(page, (290, 770, 310, 790), str(page + 1)),
            _block(page, (60, 150, 540, 400), f"unique body {page}"),
        ]))
    signatures = repeated_margin_signatures(pages)
    assert "research conference #" in signatures
    assert "#" in signatures
    assert "unique body #" not in signatures


def test_line_join_repairs_hyphenation_and_table_is_markdown():
    assert join_pdf_lines(["multi-", "modal retrieval"]) == "multimodal retrieval"
    table = markdown_table([
        ["Model", "Score"],
        ["BGE|M3", "0.82"],
    ])
    assert "| Model | Score |" in table
    assert "BGE\\|M3" in table


def test_font_and_heading_pattern_preserve_academic_sections():
    values = classify_pdf_blocks([
        _block(0, (60, 100, 400, 125), "2. Methods", size=14, bold=True),
        _block(0, (60, 140, 540, 260), "body " * 30, size=10),
    ])
    assert values[0].kind == "heading"
    assert values[0].text.startswith("##") or values[0].text.startswith("###")
    assert values[1].kind == "paragraph"

