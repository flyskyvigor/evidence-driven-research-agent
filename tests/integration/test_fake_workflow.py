import pytest
from types import SimpleNamespace
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.workflow.graph import build_research_graph
from research_agent.workflow.nodes import ResearchNodes


class NoopLLM:
    pass


class FakeMemoryManager:
    def __init__(self):
        self.recalled = False
        self.persisted = False

    def recall(self, owner_id, query):
        self.recalled = True
        return ([{"memory_type": "preference", "content": "preferred_sources=paper"}], "fake")

    def format_context(self, memories):
        return "[M1/preference] preferred_sources=paper"

    def remember_run(self, state):
        self.persisted = True
        return SimpleNamespace(memory_id="MEM-1", action="created", content_hash="hash")


@pytest.mark.integration
def test_complete_graph_retries_once_and_stops(monkeypatch):
    monkeypatch.setattr(ResearchNodes, "planner", lambda self, state: {
        "plan": {}, "analysis_dimensions": [], "round": 0, "all_evidence": [], "evidence": []
    })
    monkeypatch.setattr(ResearchNodes, "approval", lambda self, state: {"approval_status": "auto_approved"})
    monkeypatch.setattr(ResearchNodes, "retrieve", lambda self, state: {
        "round": state.get("round", 0) + 1,
        "all_evidence": [{"title": "fixture"}],
    })
    monkeypatch.setattr(ResearchNodes, "score_evidence", lambda self, state: {
        "evidence": [{"title": "fixture", "overall_score": 0.8}]
    })
    monkeypatch.setattr(ResearchNodes, "research", lambda self, state: {"researcher_output": {"claims": []}})
    monkeypatch.setattr(ResearchNodes, "critic", lambda self, state: {
        "sufficient": state["round"] >= 2,
        "stop_reason": "done" if state["round"] >= 2 else "retry",
    })
    monkeypatch.setattr(ResearchNodes, "finalize", lambda self, state: {"final_answer": "done"})

    graph = build_research_graph(NoopLLM(), checkpointer=InMemorySaver())
    result = graph.invoke(
        {"question": "fixture"},
        config={"configurable": {"thread_id": "test-thread"}},
    )
    assert result["round"] == 2
    assert result["final_answer"] == "done"


@pytest.mark.integration
def test_memory_nodes_are_in_the_formal_graph_path(monkeypatch):
    monkeypatch.setattr(ResearchNodes, "planner", lambda self, state: {
        "plan": {"memory_seen": state.get("memory_context")},
        "analysis_dimensions": [], "round": 0, "all_evidence": [], "evidence": [],
    })
    monkeypatch.setattr(ResearchNodes, "approval", lambda self, state: {"approval_status": "auto_approved"})
    monkeypatch.setattr(ResearchNodes, "retrieve", lambda self, state: {
        "round": 1, "all_evidence": [{"title": "fixture"}],
    })
    monkeypatch.setattr(ResearchNodes, "score_evidence", lambda self, state: {
        "evidence": [{"title": "fixture", "overall_score": 0.8}]
    })
    monkeypatch.setattr(ResearchNodes, "research", lambda self, state: {"researcher_output": {"claims": []}})
    monkeypatch.setattr(ResearchNodes, "critic", lambda self, state: {
        "sufficient": True, "stop_reason": "done",
    })
    monkeypatch.setattr(ResearchNodes, "finalize", lambda self, state: {
        "final_answer": "done", "verified_claims": [{"claim": "verified"}],
    })

    memory = FakeMemoryManager()
    graph = build_research_graph(
        NoopLLM(), checkpointer=InMemorySaver(), memory_manager=memory
    )
    result = graph.invoke(
        {"question": "fixture", "user_id": "user-a", "run_id": "run-a"},
        config={"configurable": {"thread_id": "memory-thread"}},
    )
    assert memory.recalled is True
    assert memory.persisted is True
    assert result["plan"]["memory_seen"].startswith("[M1/preference]")
    assert result["memory_write"]["memory_id"] == "MEM-1"
