"""不使用 LLM Judge 的基础确定性指标。"""

from __future__ import annotations

import re
from typing import Any


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence", []) if isinstance(result.get("evidence"), list) else []
    claims = result.get("verified_claims", []) if isinstance(result.get("verified_claims"), list) else []
    final_answer = str(result.get("final_answer") or "")
    cited = {int(value) for value in re.findall(r"\[E([1-9]\d*)\]", final_answer)}
    valid_citations = {value for value in cited if value <= len(evidence)}
    claims_with_support = sum(
        isinstance(item, dict) and bool(item.get("evidence_ids"))
        for item in claims
    )
    tool_results = result.get("tool_results", []) if isinstance(result.get("tool_results"), list) else []
    tool_successes = sum(
        isinstance(item, dict) and item.get("status") == "success"
        for item in tool_results
    )
    expected = case.get("expected", {})
    expected_sources = set(expected.get("required_source_types", []))
    actual_sources = {
        str(item.get("source_type")) for item in evidence if isinstance(item, dict)
    }
    local_evidence = [
        item for item in evidence
        if isinstance(item, dict) and item.get("source_type") == "local_rag"
    ]
    return {
        "case_id": case.get("id"),
        "citation_validity": len(valid_citations) / len(cited) if cited else 1.0,
        "claim_support_rate": claims_with_support / len(claims) if claims else 1.0,
        "retrieval_source_recall": (
            len(expected_sources & actual_sources) / len(expected_sources)
            if expected_sources else 1.0
        ),
        "task_completed": bool(final_answer),
        "tool_success_rate": tool_successes / len(tool_results) if tool_results else None,
        "round_limit_respected": int(result.get("round", 0) or 0) <= int(expected.get("max_rounds", 2)),
        "insufficient_detected": (
            result.get("sufficient") is False
            if expected.get("expect_insufficient")
            else None
        ),
        "conflict_detected": (
            bool(result.get("evidence_conflicts"))
            if expected.get("expect_conflict")
            else None
        ),
        "tool_failure_detected": (
            bool(result.get("tool_failures"))
            if expected.get("expect_tool_failure")
            else None
        ),
        "parent_child_traceable": (
            bool(local_evidence)
            and all(
                item.get("parent_id") and item.get("matched_child_ids")
                for item in local_evidence
            )
            if expected.get("require_parent_child_trace")
            else None
        ),
        "memory_recall_observed": (
            bool(result.get("recalled_memories"))
            if expected.get("expect_memory_recall")
            else None
        ),
        "pdf_traceable": (
            bool(local_evidence)
            and all(
                str(item.get("parser") or "").startswith("academic_pdf_pymupdf")
                and item.get("page_start")
                for item in local_evidence
            )
            if expected.get("require_pdf_trace")
            else None
        ),
        "tool_duration_ms": result.get("run_metrics", {}).get("tool_duration_ms"),
    }
