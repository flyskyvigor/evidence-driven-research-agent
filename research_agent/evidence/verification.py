"""不依赖模型的证据约束。

LLM 可以判断语义支持关系，但引用存在性、稳定 ID、独立来源数和冲突门禁应由
程序确定，避免提示词被误当成系统保证。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def assign_evidence_identity(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        value = dict(item)
        fingerprint_text = "\n".join([
            str(value.get("source_type") or ""),
            str(value.get("url") or "").strip().lower(),
            str(value.get("title") or "").strip().lower(),
            str(value.get("content") or "")[:2000],
        ])
        digest = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()
        value["content_hash"] = digest
        value["evidence_id"] = f"EV-{digest[:16]}"
        value.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
        normalized.append(value)
    return normalized


def support_metrics(
    evidence_ids: list[str],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    positions = []
    for evidence_id in evidence_ids:
        match = re.fullmatch(r"E([1-9]\d*)", str(evidence_id).upper())
        if match and int(match.group(1)) <= len(evidence):
            positions.append(int(match.group(1)) - 1)

    selected = [evidence[index] for index in sorted(set(positions))]
    independent_sources = {_source_identity(item) for item in selected}
    stable_ids = [str(item.get("evidence_id") or "") for item in selected]
    source_types = sorted({str(item.get("source_type") or "unknown") for item in selected})
    return {
        "evidence_count": len(selected),
        "independent_source_count": len(independent_sources),
        "source_types": source_types,
        "stable_evidence_ids": [item for item in stable_ids if item],
        "cross_verified": len(independent_sources) >= 2,
    }


def normalize_conflicts(
    values: Any,
    evidence_count: int,
    known_claim_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    conflicts = []
    for item in values:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or "").upper()
        if claim_id and claim_id not in known_claim_ids:
            continue
        evidence_for = _valid_citations(item.get("evidence_for"), evidence_count)
        evidence_against = _valid_citations(item.get("evidence_against"), evidence_count)
        if not evidence_for or not evidence_against:
            continue
        status = str(item.get("status") or "unresolved").lower()
        if status not in {"resolved", "unresolved"}:
            status = "unresolved"
        conflicts.append({
            "claim_id": claim_id,
            "topic": str(item.get("topic") or "").strip(),
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "status": status,
            "resolution": str(item.get("resolution") or "").strip(),
        })
    return conflicts


def _valid_citations(values: Any, evidence_count: int) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result = []
    for value in values:
        match = re.fullmatch(r"E([1-9]\d*)", str(value).strip().upper())
        if not match or int(match.group(1)) > evidence_count:
            continue
        normalized = f"E{int(match.group(1))}"
        if normalized not in result:
            result.append(normalized)
    return result


def _source_identity(item: dict[str, Any]) -> str:
    url = str(item.get("url") or "")
    hostname = (urlparse(url).hostname or "").lower()
    if hostname:
        return hostname.removeprefix("www.")
    return f"{item.get('source_type', 'unknown')}:{item.get('source_name', '')}"

