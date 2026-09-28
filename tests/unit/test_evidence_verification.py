from research_agent.evidence.verification import (
    assign_evidence_identity,
    normalize_conflicts,
    support_metrics,
)


def test_stable_identity_and_cross_source_metrics():
    evidence = assign_evidence_identity([
        {"source_type": "web", "url": "https://a.example/x", "title": "A", "content": "fact"},
        {"source_type": "paper", "url": "https://b.example/y", "title": "B", "content": "fact"},
    ])
    assert evidence[0]["evidence_id"].startswith("EV-")
    metrics = support_metrics(["E1", "E2", "E99"], evidence)
    assert metrics["cross_verified"] is True
    assert metrics["independent_source_count"] == 2


def test_conflicts_require_two_valid_sides():
    conflicts = normalize_conflicts([
        {
            "claim_id": "C1",
            "topic": "opposite results",
            "evidence_for": ["E1"],
            "evidence_against": ["E2", "E9"],
            "status": "unresolved",
        }
    ], 2, {"C1"})
    assert conflicts[0]["evidence_against"] == ["E2"]
