from research_agent.memory.store import ConversationMemoryStore


def test_sessions_are_isolated(tmp_path):
    store = ConversationMemoryStore(tmp_path / "memory.sqlite")
    store.save_turn("A", 1, "qa", "aa", {
        "turn": 1, "user_question": "qa", "result": {"final_answer": "aa"}
    })
    store.save_turn("B", 1, "qb", "ab", {
        "turn": 1, "user_question": "qb", "result": {"final_answer": "ab"}
    })
    assert store.load_session("A")[0]["user_question"] == "qa"
    assert store.context_messages("A")[0]["content"] == "qa"
    assert store.context_messages("B")[0]["content"] == "qb"
