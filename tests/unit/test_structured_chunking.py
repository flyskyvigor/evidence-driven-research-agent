from langchain_core.documents import Document

from research_agent.rag.chunking import parse_structured_units, split_structured_document


def test_structured_units_keep_heading_and_table_boundaries():
    text = """## Results

The experiment compares two retrieval strategies and reports stable results.

| Method | Recall |
| --- | --- |
| Dense | 0.70 |
| Hybrid | 0.82 |

## Limitations

The evaluation corpus is intentionally small.
"""
    units = parse_structured_units(text)
    assert [unit.kind for unit in units] == [
        "heading", "paragraph", "table", "heading", "paragraph"
    ]
    chunks = split_structured_document(
        Document(page_content=text, metadata={"page_start": 1}),
        chunk_size=150,
        chunk_overlap=20,
    )
    assert any("| Hybrid | 0.82 |" in chunk.page_content for chunk in chunks)
    assert all(chunk.metadata.get("section_title") for chunk in chunks)


def test_large_table_repeats_header_when_split_by_rows():
    rows = "\n".join(f"| row-{index} | value-{index} |" for index in range(20))
    text = f"| Name | Value |\n| --- | --- |\n{rows}"
    chunks = split_structured_document(
        Document(page_content=text, metadata={}),
        chunk_size=140,
        chunk_overlap=0,
    )
    assert len(chunks) > 1
    assert all(chunk.page_content.startswith("| Name | Value |") for chunk in chunks)

