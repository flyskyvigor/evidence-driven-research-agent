import json
import argparse
import uuid
from collections import Counter

from langgraph.types import Command

from research_agent.config import get_llm_model_path
from research_agent.llm.qwen import QwenLLM
from research_agent.workflow.graph import build_research_graph
from research_agent.memory import LongTermMemoryManager, create_sqlite_checkpointer
from research_agent.observability import configure_logging



def _arguments():
    parser = argparse.ArgumentParser(description="Evidence-driven research agent CLI")
    parser.add_argument("--session-id", default="", help="稳定会话ID；复用它可读取同一检查点")
    parser.add_argument("--user-id", default="", help="可选用户ID；复用它可跨会话召回长期记忆")
    parser.add_argument("--resume", choices=["approve", "reject"], help="恢复等待人工确认的会话")
    parser.add_argument(
        "--set-preference",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="显式保存白名单用户偏好，可重复传入",
    )
    parser.add_argument(
        "--remember-fact",
        action="append",
        default=[],
        metavar="KEY=TEXT",
        help="显式保存用户授权的长期事实，可重复传入",
    )
    parser.add_argument(
        "--forget-memory",
        action="append",
        default=[],
        metavar="MEMORY_ID",
        help="删除当前用户的一条长期记忆，可重复传入",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="只执行记忆写入/删除/过期清理，不启动研究任务",
    )
    return parser.parse_args()


def _key_value(value):
    key, separator, content = value.partition("=")
    if not separator or not key.strip() or not content.strip():
        raise ValueError("Memory values must use KEY=VALUE format")
    return key.strip(), content.strip()


def main():
    configure_logging()
    args = _arguments()
    llm = QwenLLM(get_llm_model_path())
    memory_manager = LongTermMemoryManager()
    graph = build_research_graph(
        llm,
        checkpointer=create_sqlite_checkpointer(),
        memory_manager=memory_manager,
    )
    session_id = args.session_id.strip() or uuid.uuid4().hex
    user_id = args.user_id.strip()
    config = {"configurable": {"thread_id": session_id}}

    print(f"Session ID: {session_id}")

    if (args.set_preference or args.remember_fact or args.forget_memory) and not user_id:
        raise ValueError("--user-id is required when modifying long-term memory")
    for raw in args.set_preference:
        key, value = _key_value(raw)
        memory_manager.remember_preference(user_id, key, value)
    for raw in args.remember_fact:
        key, value = _key_value(raw)
        memory_manager.remember_semantic(
            user_id, key, value, source_session_id=session_id
        )
    for memory_id in args.forget_memory:
        deleted = memory_manager.forget(user_id, memory_id)
        print(f"Forget {memory_id}: {'deleted' if deleted else 'not found'}")
    if args.memory_only:
        print(f"Expired memories purged: {memory_manager.purge_expired()}")
        return

    if args.resume:
        result = graph.invoke(
            Command(resume={"approved": args.resume == "approve", "comment": "CLI resume"}),
            config=config,
        )
    else:
        question = input("请输入你想探索的问题：\n> ")
        result = graph.invoke(
            {
                "question": question,
                "session_id": session_id,
                "user_id": user_id,
                "run_id": uuid.uuid4().hex,
            },
            config=config,
        )

    if result.get("__interrupt__"):
        print("\n===== Waiting for human approval =====")
        print(result["__interrupt__"])
        print(f"Resume with: python -m scripts.run_cli --session-id {session_id} --resume approve")
        return

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
