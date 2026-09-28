"""分层会话 Memory。

这里只保存可解释的用户/助手消息、研究记录摘要和显式用户偏好。原始网页全文、
密钥、模型隐藏推理和无边界历史不进入长期存储。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from research_agent.config import get_max_context_chars, get_memory_db_path


class ConversationMemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else get_memory_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = RLock()
        self._setup()

    def _setup(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS user_preferences (
                    session_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, preference_key)
                );
                """
            )

    def save_turn(
        self,
        session_id: str,
        turn_id: int,
        user_text: str,
        assistant_text: str,
        record: dict[str, Any],
    ) -> None:
        safe_record = _memory_safe_record(record)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO conversation_turns
                (session_id, turn_id, user_text, assistant_text, record_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    int(turn_id),
                    user_text[:4000],
                    assistant_text[:8000],
                    json.dumps(safe_record, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_session(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT record_json FROM conversation_turns
                WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?""",
                (session_id, max(1, min(int(limit), 100))),
            ).fetchall()
        return [json.loads(row["record_json"]) for row in reversed(rows)]

    def context_messages(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        max_chars = get_max_context_chars()
        with self._lock:
            rows = self._conn.execute(
                """SELECT user_text, assistant_text FROM conversation_turns
                WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?""",
                (session_id, max(1, min(int(limit), 20))),
            ).fetchall()
        messages = []
        used = 0
        for row in reversed(rows):
            for role, key in (("user", "user_text"), ("assistant", "assistant_text")):
                text = row[key]
                remaining = max_chars - used
                if remaining <= 0:
                    return messages
                text = text[:remaining]
                messages.append({"role": role, "content": text})
                used += len(text)
        return messages

    def set_preference(self, session_id: str, key: str, value: str) -> None:
        allowed = {"answer_language", "answer_style", "preferred_sources"}
        if key not in allowed:
            raise ValueError(f"Preference key is not allowed: {key}")
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO user_preferences
                (session_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, ?)""",
                (session_id, key, value[:1000], datetime.now(timezone.utc).isoformat()),
            )

    def preferences(self, session_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT preference_key, preference_value FROM user_preferences WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {row["preference_key"]: row["preference_value"] for row in rows}

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM user_preferences WHERE session_id = ?", (session_id,))


def _memory_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        "turn": record.get("turn"),
        "user_question": record.get("user_question", ""),
        "standalone_question": record.get("standalone_question", ""),
        "intent": record.get("intent", ""),
        "intent_reason": record.get("intent_reason", ""),
        "topic_anchors": record.get("topic_anchors", []),
    }
    run = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
    result["result"] = {
        "plan": run.get("plan", {}),
        "verified_claims": run.get("verified_claims", []),
        "unsupported_claims": run.get("unsupported_claims", []),
        "missing_perspectives": run.get("missing_perspectives", []),
        "evidence_conflicts": run.get("evidence_conflicts", []),
        "sufficient": run.get("sufficient"),
        "round": run.get("round", 0),
        "final_answer": str(run.get("final_answer") or "")[:8000],
    }
    return result
