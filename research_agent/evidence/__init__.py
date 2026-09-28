"""证据身份、Claim 支持度和冲突校验。"""

from research_agent.evidence.verification import (
    assign_evidence_identity,
    normalize_conflicts,
    support_metrics,
)

__all__ = ["assign_evidence_identity", "normalize_conflicts", "support_metrics"]
