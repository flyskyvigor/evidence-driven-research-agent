from langgraph.graph import END, START, StateGraph

from research_agent.workflow.nodes import ResearchNodes
from research_agent.workflow.state import ResearchState


def build_research_graph(llm, checkpointer=None, memory_manager=None):
    nodes = ResearchNodes(llm, memory_manager=memory_manager)
    builder = StateGraph(ResearchState)

    builder.add_node("load_memory", nodes.load_memory)
    builder.add_node("planner", nodes.planner)
    builder.add_node("approval", nodes.approval)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("score_evidence", nodes.score_evidence)
    builder.add_node("researcher", nodes.research)
    builder.add_node("critic", nodes.critic)
    builder.add_node("finalize", nodes.finalize)
    builder.add_node("persist_memory", nodes.persist_memory)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "planner")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    builder.add_edge("retrieve", "score_evidence")
    builder.add_edge("score_evidence", "researcher")
    builder.add_edge("researcher", "critic")

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retrieve": "retrieve",
            "finalize": "finalize"
        }
    )

    builder.add_edge("finalize", "persist_memory")
    builder.add_edge("persist_memory", END)

    return builder.compile(checkpointer=checkpointer)


def route_after_approval(state):
    return "retrieve" if state.get("approval_status") != "rejected" else "finalize"


def route_after_critic(state):
    if not state.get("sufficient", False) and state.get("round", 0) < 2:
        return "retrieve"

    return "finalize"
