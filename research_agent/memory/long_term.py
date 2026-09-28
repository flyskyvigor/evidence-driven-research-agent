"""Opt-in long-term memory with explicit ownership and retention rules.

Long-term memory is not a second evidence source. Recalled items are supplied
only to the Planner as historical constraints and query hints; factual output
must still pass the normal retrieval and Claim-to-Evidence gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol

from research_agent.config import (
    get_memory_db_path,
    get_memory_recall_limit,
    get_memory_ttl_days,
)
from research_agent.rag.hybrid import tokenize_for_search


logger = logging.getLogger(__name__)
ALLOWED_MEMORY_TYPES = {"preference", "semantic", "episodic"}
ALLOWED_PREFERENCE_KEYS = {"answer_language", "answer_style", "preferred_sources"}


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class MemoryWriteResult:
    memory_id: str
    action: str
    content_hash: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _content_hash(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _lexical_overlap(query: str, content: str) -> float:
    query_tokens = set(tokenize_for_search(query))
    content_tokens = set(tokenize_for_search(content))
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


class LongTermMemoryManager:
    """SQLite memory store plus semantic/recency/importance retrieval policy."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_factory: Callable[[], EmbeddingProvider] | None = None,
    ) -> None:
        self.path = Path(path) if path else get_memory_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = RLock()
        self._embedding_provider = embedding_provider
        self._embedding_factory = embedding_factory or _default_embedding_factory
        self._embedding_failed = False
        self._setup()

    def _setup(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    source_session_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    expires_at TEXT,
                    UNIQUE(owner_id, memory_type, memory_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_owner_type
                    ON long_term_memories(owner_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_owner_hash
                    ON long_term_memories(owner_id, memory_type, content_hash);
                CREATE INDEX IF NOT EXISTS idx_memory_expiry
                    ON long_term_memories(expires_at);
                """
            )

    def _embed(self, text: str) -> list[float]:
        if self._embedding_failed:
            return []
        try:
            if self._embedding_provider is None:
                self._embedding_provider = self._embedding_factory()
            values = self._embedding_provider.embed_query(text)
            return [float(value) for value in values]
        except Exception:
            self._embedding_failed = True
            logger.exception("long_term_memory_embedding_failed")
            return []

    @staticmethod
    def _validate_owner(owner_id: str) -> str:
        value = str(owner_id or "").strip()
        if not value or len(value) > 200:
            raise ValueError("owner_id must contain 1 to 200 characters")
        return value

    def remember(
        self,
        *,
        owner_id: str,
        memory_type: str,
        memory_key: str,
        content: str,
        importance: float = 0.5,
        source_session_id: str = "",
        metadata: dict[str, Any] | None = None,
        ttl_days: int | None = None,
        now: datetime | None = None,
    ) -> MemoryWriteResult:
        owner_id = self._validate_owner(owner_id)
        memory_type = str(memory_type or "").strip().lower()
        if memory_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(f"Unsupported memory_type: {memory_type}")
        memory_key = str(memory_key or "").strip()
        if not memory_key or len(memory_key) > 300:
            raise ValueError("memory_key must contain 1 to 300 characters")
        content = re.sub(r"\s+", " ", str(content or "")).strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        content = content[:6000]
        importance = max(0.0, min(float(importance), 1.0))
        digest = _content_hash(content)
        timestamp = now or _now()
        ttl = get_memory_ttl_days() if ttl_days is None else int(ttl_days)
        expires_at = None if memory_type == "preference" else _iso(
            timestamp + timedelta(days=max(1, ttl))
        )
        embedding = [] if memory_type == "preference" else self._embed(content)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._lock, self._conn:
            duplicate = self._conn.execute(
                """SELECT memory_id FROM long_term_memories
                WHERE owner_id = ? AND memory_type = ? AND content_hash = ?
                LIMIT 1""",
                (owner_id, memory_type, digest),
            ).fetchone()
            if duplicate:
                self._conn.execute(
                    """UPDATE long_term_memories
                    SET updated_at = ?, expires_at = ?, importance = MAX(importance, ?)
                    WHERE memory_id = ?""",
                    (_iso(timestamp), expires_at, importance, duplicate["memory_id"]),
                )
                return MemoryWriteResult(
                    memory_id=duplicate["memory_id"],
                    action="deduplicated",
                    content_hash=digest,
                )

            existing = self._conn.execute(
                """SELECT memory_id FROM long_term_memories
                WHERE owner_id = ? AND memory_type = ? AND memory_key = ?""",
                (owner_id, memory_type, memory_key),
            ).fetchone()
            memory_id = existing["memory_id"] if existing else f"MEM-{uuid.uuid4().hex[:20]}"
            if existing:
                self._conn.execute(
                    """UPDATE long_term_memories SET
                    content = ?, content_hash = ?, embedding_json = ?, importance = ?,
                    source_session_id = ?, metadata_json = ?, updated_at = ?,
                    last_accessed_at = ?, expires_at = ? WHERE memory_id = ?""",
                    (
                        content, digest, json.dumps(embedding), importance,
                        source_session_id[:200], metadata_json, _iso(timestamp),
                        _iso(timestamp), expires_at, memory_id,
                    ),
                )
                action = "updated"
            else:
                self._conn.execute(
                    """INSERT INTO long_term_memories (
                    memory_id, owner_id, memory_type, memory_key, content,
                    content_hash, embedding_json, importance, source_session_id,
                    metadata_json, created_at, updated_at, last_accessed_at,
                    access_count, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (
                        memory_id, owner_id, memory_type, memory_key, content,
                        digest, json.dumps(embedding), importance,
                        source_session_id[:200], metadata_json, _iso(timestamp),
                        _iso(timestamp), _iso(timestamp), expires_at,
                    ),
                )
                action = "created"
        return MemoryWriteResult(memory_id, action, digest)

    def remember_preference(
        self, owner_id: str, key: str, value: str
    ) -> MemoryWriteResult:
        if key not in ALLOWED_PREFERENCE_KEYS:
            raise ValueError(f"Preference key is not allowed: {key}")
        value = str(value or "").strip()
        if not value:
            raise ValueError("Preference value cannot be empty")
        return self.remember(
            owner_id=owner_id,
            memory_type="preference",
            memory_key=key,
            content=f"{key}={value[:1000]}",
            importance=1.0,
            metadata={"preference_key": key, "preference_value": value[:1000]},
        )

    def remember_semantic(
        self,
        owner_id: str,
        key: str,
        content: str,
        *,
        source_session_id: str = "",
        importance: float = 0.7,
    ) -> MemoryWriteResult:
        """Store a user-authorized durable fact; never called from hidden reasoning."""
        return self.remember(
            owner_id=owner_id,
            memory_type="semantic",
            memory_key=key,
            content=content,
            source_session_id=source_session_id,
            importance=importance,
        )

    def remember_run(self, state: dict[str, Any]) -> MemoryWriteResult | None:
        owner_id = str(state.get("user_id") or "").strip()
        verified_claims = state.get("verified_claims") or []
        if not owner_id or not verified_claims:
            return None
        claim_texts = [
            str(item.get("claim") or "").strip()
            for item in verified_claims
            if isinstance(item, dict) and item.get("claim")
        ][:8]
        if not claim_texts:
            return None
        question = str(state.get("question") or "")[:1000]
        content = f"历史研究问题：{question}；已核验结论：" + "；".join(claim_texts)
        return self.remember(
            owner_id=owner_id,
            memory_type="episodic",
            memory_key=f"run:{state.get('run_id') or _content_hash(question)[:16]}",
            content=content,
            importance=0.8 if state.get("sufficient") else 0.6,
            source_session_id=str(state.get("session_id") or ""),
            metadata={
                "question": question,
                "claim_count": len(claim_texts),
                "sufficient": bool(state.get("sufficient")),
            },
        )

    def recall(
        self,
        owner_id: str,
        query: str,
        *,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        owner_id = self._validate_owner(owner_id)
        timestamp = now or _now()
        limit = max(1, min(int(limit or get_memory_recall_limit()), 20))
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM long_term_memories
                WHERE owner_id = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC LIMIT 2000""",
                (owner_id, _iso(timestamp)),
            ).fetchall()
        if not rows:
            return [], "empty"

        has_retrievable_content = any(
            row["memory_type"] != "preference" for row in rows
        )
        query_embedding = self._embed(query) if has_retrievable_content else []
        if not has_retrievable_content:
            mode = "preferences_only"
        else:
            mode = (
                "semantic+lexical+recency+importance"
                if query_embedding else "lexical_fallback"
            )
        ranked: list[dict[str, Any]] = []
        for row in rows:
            memory_type = row["memory_type"]
            try:
                embedding = [float(value) for value in json.loads(row["embedding_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                embedding = []
            semantic = _cosine(query_embedding, embedding)
            lexical = _lexical_overlap(query, row["content"])
            updated_at = _parse_time(row["updated_at"]) or timestamp
            age_days = max(0.0, (timestamp - updated_at).total_seconds() / 86400.0)
            recency = math.exp(-age_days / 90.0)
            importance = max(0.0, min(float(row["importance"]), 1.0))
            if memory_type == "preference":
                score = 1.0
            else:
                score = (
                    0.55 * semantic
                    + 0.20 * lexical
                    + 0.15 * importance
                    + 0.10 * recency
                )
            ranked.append({
                "memory_id": row["memory_id"],
                "memory_type": memory_type,
                "memory_key": row["memory_key"],
                "content": row["content"],
                "importance": importance,
                "updated_at": row["updated_at"],
                "expires_at": row["expires_at"],
                "score": round(score, 6),
                "score_components": {
                    "semantic": round(semantic, 6),
                    "lexical": round(lexical, 6),
                    "recency": round(recency, 6),
                    "importance": round(importance, 6),
                },
                "metadata": json.loads(row["metadata_json"] or "{}"),
            })
        ranked.sort(
            key=lambda item: (
                0 if item["memory_type"] == "preference" else 1,
                -item["score"],
                item["memory_id"],
            )
        )
        selected = ranked[:limit]
        if selected:
            memory_ids = [item["memory_id"] for item in selected]
            placeholders = ",".join("?" for _ in memory_ids)
            with self._lock, self._conn:
                self._conn.execute(
                    f"""UPDATE long_term_memories
                    SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE memory_id IN ({placeholders})""",
                    [_iso(timestamp), *memory_ids],
                )
        return selected, mode

    @staticmethod
    def format_context(memories: list[dict[str, Any]], max_chars: int = 4000) -> str:
        lines: list[str] = []
        used = 0
        for index, item in enumerate(memories, 1):
            line = f"[M{index}/{item.get('memory_type')}] {item.get('content', '')}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def forget(self, owner_id: str, memory_id: str) -> bool:
        owner_id = self._validate_owner(owner_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM long_term_memories WHERE owner_id = ? AND memory_id = ?",
                (owner_id, memory_id),
            )
        return cursor.rowcount > 0

    def delete_owner(self, owner_id: str) -> int:
        owner_id = self._validate_owner(owner_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM long_term_memories WHERE owner_id = ?", (owner_id,)
            )
        return cursor.rowcount

    def purge_expired(self, now: datetime | None = None) -> int:
        timestamp = now or _now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM long_term_memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (_iso(timestamp),),
            )
        return cursor.rowcount


def _default_embedding_factory() -> EmbeddingProvider:
    # Lazy import prevents Memory from loading BGE during module import/tests.
    from research_agent.rag.knowledge_base import create_embeddings

    return create_embeddings()

