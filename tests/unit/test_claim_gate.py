from research_agent.workflow.nodes import ResearchNodes


def test_unsupported_and_unresolved_conflict_never_enter_verified_claims():
    state = {
        "researcher_output": {"claims": [
            {"claim_id": "C1", "claim": "supported", "evidence_ids": ["E1"]},
            {"claim_id": "C2", "claim": "conflicted", "evidence_ids": ["E2"]},
            {"claim_id": "C3", "claim": "unsupported", "evidence_ids": ["E1"]},
        ]},
        "evidence": [
            {"evidence_id": "EV-1", "url": "https://a.example"},
            {"evidence_id": "EV-2", "url": "https://b.example"},
        ],
        "claim_reviews": [
            {"claim_id": "C1", "verdict": "supported", "valid_evidence_ids": ["E1"]},
            {"claim_id": "C2", "verdict": "supported", "valid_evidence_ids": ["E2"]},
            {"claim_id": "C3", "verdict": "unsupported", "valid_evidence_ids": []},
        ],
        "unsupported_claims": [{"claim_id": "C3"}],
        "evidence_conflicts": [{"claim_id": "C2", "status": "unresolved"}],
    }
    assert [item["claim_id"] for item in ResearchNodes._verified_claims(state)] == ["C1"]
