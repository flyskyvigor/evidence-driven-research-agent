from datetime import datetime, timedelta, timezone

from research_agent.memory.long_term import LongTermMemoryManager


class FakeEmbeddings:
    def embed_query(self, text):
        lowered = text.lower()
        return [
            float(lowered.count("agent") + lowered.count("智能体")),
            float(lowered.count("rag") + lowered.count("检索")),
            1.0,
        ]


def test_long_term_memory_isolated_by_owner_and_semantically_recalled(tmp_path):
    manager = LongTermMemoryManager(
        tmp_path / "memory.sqlite", embedding_provider=FakeEmbeddings()
    )
    manager.remember_semantic("user-a", "agent-project", "Agent 工具调用需要参数校验")
    manager.remember_semantic("user-b", "rag-project", "RAG 使用父子索引")

    memories, mode = manager.recall("user-a", "Agent 如何调用工具")
    assert mode == "semantic+lexical+recency+importance"
    assert [item["memory_key"] for item in memories] == ["agent-project"]
    assert all(item["memory_key"] != "rag-project" for item in memories)


def test_same_key_updates_conflicting_value_and_same_content_is_deduplicated(tmp_path):
    manager = LongTermMemoryManager(
        tmp_path / "memory.sqlite", embedding_provider=FakeEmbeddings()
    )
    created = manager.remember_preference("user-a", "answer_style", "简洁")
    updated = manager.remember_preference("user-a", "answer_style", "详细")
    duplicate = manager.remember(
        owner_id="user-a",
        memory_type="preference",
        memory_key="another-key",
        content="answer_style=详细",
        importance=1.0,
    )

    assert created.action == "created"
    assert updated.action == "updated"
    assert updated.memory_id == created.memory_id
    assert duplicate.action == "deduplicated"
    assert duplicate.memory_id == created.memory_id
    memories, _ = manager.recall("user-a", "任意问题")
    assert memories[0]["content"] == "answer_style=详细"


def test_expired_memory_is_not_recalled_and_can_be_purged(tmp_path):
    manager = LongTermMemoryManager(
        tmp_path / "memory.sqlite", embedding_provider=FakeEmbeddings()
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manager.remember(
        owner_id="user-a",
        memory_type="episodic",
        memory_key="old-run",
        content="一条会过期的 Agent 研究记录",
        ttl_days=1,
        now=start,
    )
    memories, _ = manager.recall(
        "user-a", "Agent", now=start + timedelta(days=2)
    )
    assert memories == []
    assert manager.purge_expired(now=start + timedelta(days=2)) == 1


def test_run_memory_only_contains_verified_summary_and_is_idempotent(tmp_path):
    manager = LongTermMemoryManager(
        tmp_path / "memory.sqlite", embedding_provider=FakeEmbeddings()
    )
    state = {
        "user_id": "user-a",
        "session_id": "session-a",
        "run_id": "run-1",
        "question": "如何设计 Agent？",
        "verified_claims": [{"claim": "工具参数需要在执行前校验"}],
        "all_evidence": [{"content": "不应进入长期记忆的网页全文"}],
        "sufficient": True,
    }
    first = manager.remember_run(state)
    second = manager.remember_run(state)
    memories, _ = manager.recall("user-a", "工具参数")

    assert first is not None and first.action == "created"
    assert second is not None and second.action == "deduplicated"
    assert "不应进入长期记忆的网页全文" not in memories[0]["content"]
    assert "工具参数需要在执行前校验" in memories[0]["content"]
    assert manager.forget("user-a", first.memory_id) is True
    assert manager.recall("user-a", "工具参数")[0] == []

