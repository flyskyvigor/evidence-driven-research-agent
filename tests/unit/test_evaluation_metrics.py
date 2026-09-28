from research_agent.evaluation.metrics import evaluate_case


def test_parent_child_and_memory_expectations_are_measured():
    case = {
        "id": "advanced-rag-memory",
        "expected": {
            "require_parent_child_trace": True,
            "expect_memory_recall": True,
            "require_pdf_trace": True,
        },
    }
    result = {
        "evidence": [{
            "source_type": "local_rag",
            "parent_id": "PAR-1",
            "matched_child_ids": ["CHD-1"],
            "parser": "academic_pdf_pymupdf_v1",
            "page_start": 2,
        }],
        "recalled_memories": [{"memory_id": "MEM-1"}],
        "final_answer": "完成",
    }
    metrics = evaluate_case(case, result)
    assert metrics["parent_child_traceable"] is True
    assert metrics["memory_recall_observed"] is True
    assert metrics["pdf_traceable"] is True

