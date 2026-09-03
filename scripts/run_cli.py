import json
from collections import Counter

from research_agent.config import get_llm_model_path
from research_agent.llm.qwen import QwenLLM
from research_agent.workflow.graph import build_research_graph



def main():
    llm = QwenLLM(get_llm_model_path())
    graph = build_research_graph(llm)

    question = input("请输入你想探索的问题：\n> ")
    result = graph.invoke({"question": question})

    print("\n===== Planner =====")
    print(
        json.dumps(
            result.get("plan", {}),
            ensure_ascii=False,
            indent=2
        )
    )

    print("\n===== Workflow =====")
    print(
        "Rounds:",
        result.get("round", 0),
        "| Sufficient:",
        result.get("sufficient", False)
    )

    print("\n===== Researcher Draft =====")
    print(result.get("draft_answer", ""))

    all_source_counts = Counter(
        item.get("source_type", "unknown")
        for item in result.get("all_evidence", [])
    )

    selected_source_counts = Counter(
        item.get("source_type", "unknown")
        for item in result.get("evidence", [])
    )

    print("\n===== Evidence Source Stats =====")
    print("Retrieved:", dict(all_source_counts))
    print("Selected :", dict(selected_source_counts))

    print(
        f"\n===== Evidence "
        f"({len(result['evidence'])}/{len(result['all_evidence'])}) ====="
    )
    for i, item in enumerate(result["evidence"], 1):
        evidence_type = "正文" if item.get("is_full_text") else "摘要"

        print(
            f"[{i}] [{item.get('source_type', '')}] "
            f"[{evidence_type}] "
            f"Score={item.get('overall_score', 0):.3f} | "
            f"Rel={item.get('relevance', 0):.2f} | "
            f"Sem={item.get('semantic_relevance', 0):.2f} | "
            f"Auth={item.get('authority', 0):.2f} | "
            f"{item.get('title', '')}"
        )

        if item.get("url"):
            print(item["url"])

    print("\n===== Critic =====")
    print(result.get("critique", ""))

    if result.get("unsupported_claims"):
        print("\nUnsupported Claims:")
        for item in result["unsupported_claims"]:
            print("-", item)

    if result.get("missing_perspectives"):
        print("\nMissing Perspectives:")
        for item in result["missing_perspectives"]:
            print("-", item)

    print("\n===== Verified Claims =====")
    verified_claims = result.get(
        "verified_claims",
        []
    )
    if not verified_claims:
        print("None")
    else:
        for item in verified_claims:
            print(
                f"{item.get('claim_id', '')}: "
                f"{item.get('claim', '')}"
            )

    print("\n===== Final Answer =====")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
