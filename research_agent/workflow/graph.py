from langgraph.graph import END, START, StateGraph

from research_agent.workflow.nodes import ResearchNodes
from research_agent.workflow.state import ResearchState


def build_research_graph(llm):
    nodes = ResearchNodes(llm)
    builder = StateGraph(ResearchState)

    builder.add_node("planner", nodes.planner)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("score_evidence", nodes.score_evidence)
    builder.add_node("researcher", nodes.research)
    builder.add_node("critic", nodes.critic)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieve")
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

    builder.add_edge("finalize", END)

    return builder.compile()


def route_after_critic(state):
    if not state.get("sufficient", False) and state.get("round", 0) < 2:
        return "retrieve"

    return "finalize"
