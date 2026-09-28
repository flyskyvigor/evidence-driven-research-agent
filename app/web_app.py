from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from textwrap import dedent
from threading import Lock
from typing import Any

import gradio as gr

from research_agent.config import get_llm_model_path
from research_agent.llm.qwen import QwenLLM
from research_agent.workflow.graph import build_research_graph
from research_agent.memory import (
    ConversationMemoryStore,
    LongTermMemoryManager,
    create_sqlite_checkpointer,
)
from research_agent.observability import configure_logging


logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 3
MAX_UI_TURNS = 20
MAX_ASSISTANT_CONTEXT_CHARS = 1800
SOURCE_ORDER = ("web", "github", "paper", "local_rag")
SOURCE_LABELS = {
    "web": "网页",
    "github": "GitHub",
    "paper": "学术论文",
    "local_rag": "本地知识库",
}
EVIDENCE_HEADERS = [
    "证据编号",
    "来源类型",
    "标题",
    "综合评分",
    "相关度",
    "语义相关度",
    "来源权威性",
    "内容类型",
    "是否引用",
    "URL",
]

NODE_NAMES = {
    "load_memory": "长期记忆召回",
    "planner": "研究规划",
    "retrieve": "多源检索",
    "score_evidence": "证据评分",
    "researcher": "研究员分析",
    "critic": "审查员评审",
    "finalize": "最终结论整理",
    "persist_memory": "研究摘要入库",
}


def build_research_backend():
    """初始化并编译唯一一份研究后端。"""
    llm = QwenLLM(get_llm_model_path())
    long_term_memory = LongTermMemoryManager()
    graph = build_research_graph(
        llm,
        checkpointer=create_sqlite_checkpointer(),
        memory_manager=long_term_memory,
    )
    return llm, graph, ConversationMemoryStore()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def _recent_dialogue(history: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """从 messages 历史中提取最近的完整问答，跳过失败的研究轮次。"""
    pairs = []
    pending_user = None

    for message in history:
        role = message.get("role")
        text = _message_text(message).strip()
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user:
            if text and not text.startswith("研究失败："):
                pairs.append((pending_user, text))
            pending_user = None

    return pairs[-MAX_HISTORY_TURNS:]


ANCHOR_GROUPS = {
    "大语言模型": ("大语言模型", "large language model", "llm", "llms"),
    "Agent": ("agent", "agents", "智能体"),
    "Reflection": ("reflection", "reflexion", "self-reflection", "自我反思"),
    "LangGraph": ("langgraph",),
    "RAG": ("rag", "检索增强生成"),
    "reranker": ("reranker", "re-ranker", "重排序器"),
    "AutoGen": ("autogen",),
    "CrewAI": ("crewai",),
}
GENERIC_ANCHOR_EXCLUSIONS = {
    "github", "web", "paper", "research", "researcher", "critic",
    "user", "assistant", "json", "evidence", "claim"
}


def _parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _contains_anchor_term(text: str, term: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", term):
        return term in text
    return bool(re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
        text
    ))


def extract_topic_anchors(text: str) -> list[str]:
    """提取少量技术实体，用于确定性检查追问改写是否丢失主题。"""
    lowered = (text or "").lower()
    anchors = []
    matched_variants = set()
    for anchor, variants in ANCHOR_GROUPS.items():
        if any(_contains_anchor_term(lowered, term) for term in variants):
            anchors.append(anchor)
            matched_variants.update(variants)

    for token in re.findall(r"\b[A-Z][A-Za-z0-9.+#_-]{2,}\b", text or ""):
        lowered_token = token.lower()
        if (
            lowered_token not in GENERIC_ANCHOR_EXCLUSIONS
            and not any(lowered_token in variant for variant in matched_variants)
            and token not in anchors
        ):
            anchors.append(token)
    return anchors[:8]


def _anchor_is_present(text: str, anchor: str) -> bool:
    lowered = (text or "").lower()
    variants = ANCHOR_GROUPS.get(anchor, (anchor.lower(),))
    return any(_contains_anchor_term(lowered, term) for term in variants)


def classify_followup_intent(
    llm: QwenLLM,
    question: str,
    previous_record: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, str]:
    """将后续输入路由为澄清、深化研究或新主题。"""
    if not previous_record:
        return {"intent": "new_topic", "reason": "当前是首轮研究"}

    clarify_markers = (
        "什么意思", "怎么理解", "为什么这么说", "这句话是什么意思",
        "具体指什么", "具体是什么", "支持了什么"
    )
    if any(marker in question for marker in clarify_markers) or re.search(
        r"\bE[1-9]\d*\b.*(?:支持|说明|意味着)",
        question,
        re.IGNORECASE
    ):
        return {"intent": "clarify_previous", "reason": "规则识别为上一轮内容解释"}

    previous_anchors = set(previous_record.get("topic_anchors", []))
    if not previous_anchors:
        previous_anchors = set(extract_topic_anchors(
            previous_record.get("standalone_question", "")
        ))
    current_anchors = set(extract_topic_anchors(question))
    distinctive_previous = previous_anchors - {"Agent", "大语言模型"}
    distinctive_current = current_anchors - {"Agent", "大语言模型"}
    if distinctive_current and distinctive_previous.isdisjoint(distinctive_current):
        return {"intent": "new_topic", "reason": "当前问题出现了不同的核心技术实体"}

    previous_result = previous_record.get("result", {})
    if not isinstance(previous_result, dict):
        previous_result = {}
    recent_dialogue = _recent_dialogue(history)[-2:]
    prompt = dedent(
        f"""\
        你是多轮研究对话的意图路由器。只输出JSON。

        上一轮用户问题：{previous_record.get('user_question', '')}
        上一轮实际研究问题：{previous_record.get('standalone_question', '')}
        上一轮已验证主张：{json.dumps(previous_result.get('verified_claims', []), ensure_ascii=False)}
        最近对话：{json.dumps(recent_dialogue, ensure_ascii=False)}
        当前问题：{question}

        intent只能是：
        - clarify_previous：解释上一轮术语、句子、Claim、引用或“为什么这么说”，不需要新证据。
        - deepen_previous：仍围绕上一主题，但要求新证据、更多论文、GitHub实现、反例或进一步比较。
        - new_topic：明确切换到新的独立研究对象。

        输出：{{"intent": "clarify_previous|deepen_previous|new_topic", "reason": "简短理由"}}
        """
    )

    try:
        parsed = _parse_json_object(llm.generate(prompt, max_new_tokens=180))
    except Exception:
        logger.exception("Follow-up intent classification failed")
        parsed = None

    if parsed and parsed.get("intent") in {
        "clarify_previous", "deepen_previous", "new_topic"
    }:
        return {
            "intent": parsed["intent"],
            "reason": str(parsed.get("reason") or "").strip()
        }

    return {"intent": "deepen_previous", "reason": "分类失败时保守保持上一轮主题"}


def rewrite_followup(
    llm: QwenLLM,
    question: str,
    history: list[dict[str, Any]],
    previous_record: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """使用研究记录改写追问，并确定性检查核心主题锚点。"""
    if not previous_record:
        return question, extract_topic_anchors(question)

    previous_result = previous_record.get("result", {})
    if not isinstance(previous_result, dict):
        previous_result = {}
    previous_question = str(previous_record.get("standalone_question") or "").strip()
    anchors = previous_record.get("topic_anchors", [])
    if not isinstance(anchors, list) or not anchors:
        anchors = extract_topic_anchors(previous_question)

    current_anchors = extract_topic_anchors(question)
    required_anchors = current_anchors or anchors
    recent_dialogue = _recent_dialogue(history)
    history_parts = []
    for index, (user_text, assistant_text) in enumerate(recent_dialogue, 1):
        history_parts.append(
            f"第{index}轮 User:\n{user_text}\n\n"
            f"第{index}轮 Assistant:\n"
            f"{assistant_text[:MAX_ASSISTANT_CONTEXT_CHARS]}"
        )

    prompt = dedent(
        f"""\
        你是研究问题改写器。将当前追问改写为无需阅读历史也能理解的自包含研究问题。

        上一轮用户问题：{previous_record.get('user_question', '')}
        上一轮实际研究问题：{previous_question}
        上一轮核心主题锚点：{json.dumps(anchors, ensure_ascii=False)}
        上一轮已验证主张：{json.dumps(previous_result.get('verified_claims', []), ensure_ascii=False)}
        最近对话：
        {chr(10).join(history_parts)}
        当前追问：{question}

        规则：
        1. 只改写问题，不回答问题，不添加历史中不存在的事实。
        2. 必须解析“它、这个、这种方式、这里”等指代。
        3. 当前追问没有明确切换研究对象时，必须保留上一轮核心研究对象和技术实体。
        4. 不得把技术领域概念改写成同名的日常生活、心理、教育或其他领域概念。
        5. 只输出standalone question，不要标题、解释、引号或前缀。
        """
    )

    try:
        rewritten = llm.generate(prompt, max_new_tokens=300).strip()
    except Exception:
        logger.exception("Follow-up rewriting failed; using anchored fallback")
        rewritten = ""

    for prefix in ("Standalone question:", "standalone question:", "改写后的问题："):
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
    rewritten = rewritten.strip(" \t\r\n\"'“”")

    missing_anchors = [
        anchor for anchor in required_anchors
        if not _anchor_is_present(rewritten, anchor)
    ]
    if not rewritten or missing_anchors:
        rewritten = f"{previous_question}\n\n在这一研究主题下，进一步回答：{question}"

    return rewritten, anchors


def answer_clarification_from_record(
    llm: QwenLLM,
    previous_record: dict[str, Any],
    question: str,
) -> str:
    """仅使用上一轮Claims与Evidence解释，不触发任何外部检索。"""
    result = previous_record.get("result", {})
    if not isinstance(result, dict):
        result = {}
    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    verified_claims = result.get("verified_claims", [])
    if not isinstance(verified_claims, list):
        verified_claims = []
    if not evidence and not verified_claims:
        return (
            "上一轮证据不足以具体说明这一点。"
            "如需要，我可以继续检索这一点的专项证据。"
        )

    evidence_parts = []
    for index, item in enumerate(evidence[:12], 1):
        if not isinstance(item, dict):
            continue
        evidence_parts.append(
            f"[E{index}] {item.get('title') or item.get('source_name', '')}\n"
            f"类型：{item.get('source_type', '')}\n"
            f"内容：{str(item.get('content') or '')[:1400]}"
        )

    prompt = dedent(
        f"""\
        你正在解释上一轮技术研究结果，不是在开始新的研究。请直接回答当前澄清问题。

        上一轮研究问题：{previous_record.get('standalone_question', '')}
        上一轮最终回答：{str(result.get('final_answer') or '')[:6000]}
        已验证主张：{json.dumps(verified_claims, ensure_ascii=False)}
        证据不足的主张：{json.dumps(result.get('unsupported_claims', []), ensure_ascii=False)}
        缺失视角：{json.dumps(result.get('missing_perspectives', []), ensure_ascii=False)}
        上一轮证据：
        {chr(10).join(evidence_parts)}

        当前澄清问题：{question}

        规则：
        1. 严格保持上一轮研究领域，只能根据上述结果、Claims和Evidence解释。
        2. 不得引入新事实、新机制或外部知识；事实说明使用上一轮真实Evidence ID引用。
        3. 不得把技术术语漂移到日常生活、心理、教育等其他语境。
        4. 如果证据不足，明确回答“上一轮证据不足以具体说明这一点。”
        5. 证据不足时可在结尾说明“如需要，我可以继续检索这一点的专项证据。”
        """
    )

    try:
        answer = llm.generate(prompt, max_new_tokens=900).strip()
    except Exception:
        logger.exception("Clarification generation failed")
        answer = ""
    if not answer:
        return (
            "上一轮证据不足以具体说明这一点。"
            "如需要，我可以继续检索这一点的专项证据。"
        )

    evidence_count = len(evidence)
    return re.sub(
        r"\[E(\d+)\]",
        lambda match: (
            match.group(0)
            if 1 <= int(match.group(1)) <= evidence_count
            else ""
        ),
        answer
    ).strip()


def _source_counts(result: dict[str, Any], key: str) -> Counter:
    items = result.get(key, [])
    if not isinstance(items, list):
        return Counter()
    return Counter(
        item.get("source_type", "unknown")
        for item in items
        if isinstance(item, dict)
    )


def _retrieval_summary(result: dict[str, Any]) -> str:
    counts = _source_counts(result, "all_evidence")
    return "，".join(
        f"{SOURCE_LABELS[source]}：{counts[source]}"
        for source in SOURCE_ORDER
    )


def _researcher_claims(result: dict[str, Any]) -> list[Any]:
    researcher_output = result.get("researcher_output", {})
    if not isinstance(researcher_output, dict):
        return []
    claims = researcher_output.get("claims", [])
    return claims if isinstance(claims, list) else []


def _completed_progress_line(
    node: str,
    event_round: int,
    result: dict[str, Any],
) -> str:
    if node == "planner":
        return "✓ 已完成问题解析与研究规划"
    if node == "retrieve":
        return f"✓ 已完成第{event_round}轮多源检索（累计：{_retrieval_summary(result)}）"
    if node == "score_evidence":
        selected_count = len(result.get("evidence", []) or [])
        return f"✓ 已完成第{event_round}轮证据评分与筛选（{selected_count}条）"
    if node == "researcher":
        return (
            f"✓ 第{event_round}轮研究员已形成"
            f"{len(_researcher_claims(result))}条候选主张"
        )
    if node == "critic":
        needs_retry = not result.get("sufficient", False) and event_round < 2
        return (
            f"⚠ 第{event_round}轮审查发现证据缺口"
            if needs_retry
            else f"✓ 第{event_round}轮审查员已完成证据充分性评审"
        )
    if node == "finalize":
        return "✓ 研究完成"
    return f"✓ 已完成{NODE_NAMES.get(node, node)}"


def _progress_markdown(
    completed_lines: list[str],
    current_node: str,
    result: dict[str, Any],
) -> str:
    """根据已完成节点生成可读的中文研究进度。"""
    lines = ["### 当前研究进度", "", *completed_lines]

    current_round = int(result.get("round", 0) or 0)
    if current_node == "planner":
        active = "● 正在执行网页、GitHub、学术论文和本地知识库检索……"
    elif current_node == "retrieve":
        active = (
            f"● 已获取{len(result.get('all_evidence', []) or [])}条候选证据，"
            "正在进行质量评分……"
        )
    elif current_node == "score_evidence":
        active = "● 研究员正在形成候选结论……"
    elif current_node == "researcher":
        active = "● 审查员正在进行证据充分性评审……"
    elif current_node == "critic" and (
        not result.get("sufficient", False) and current_round < 2
    ):
        active = f"● 正在进行第{current_round + 1}轮定向补充检索……"
    elif current_node == "critic":
        active = "● 正在整理最终结论……"
    else:
        active = ""

    if active:
        lines.extend(["", active])
    return "\n".join(lines)


def _chat_status(current_node: str, result: dict[str, Any]) -> str:
    """生成 Chatbot 中随节点变化的简短状态。"""
    current_round = int(result.get("round", 0) or 0)
    if current_node == "planner":
        return "正在执行网页、GitHub、学术论文和本地知识库检索……"
    if current_node == "retrieve":
        return (
            f"已获取{len(result.get('all_evidence', []) or [])}条候选证据，"
            "正在进行质量评分……"
        )
    if current_node == "score_evidence":
        return "研究员正在形成候选结论……"
    if current_node == "researcher":
        return "审查员正在进行证据审查……"
    if current_node == "critic" and (
        not result.get("sufficient", False) and current_round < 2
    ):
        return f"发现证据缺口，正在进行第{current_round + 1}轮定向补充检索……"
    if current_node == "critic":
        return "正在整理最终结论……"
    if current_node == "finalize":
        return "研究完成，正在呈现最终结论……"
    return f"正在执行{NODE_NAMES.get(current_node, current_node)}……"


def run_research_stream(
    graph: Any,
    standalone_question: str,
    session_id: str,
    user_id: str = "",
):
    """流式累积 LangGraph 节点更新，并逐节点产出中文进度。"""
    result: dict[str, Any] = {"question": standalone_question}
    completed_lines: list[str] = []

    for update in graph.stream(
        {
            "question": standalone_question,
            "session_id": session_id,
            "user_id": user_id,
            "run_id": uuid.uuid4().hex,
        },
        config={"configurable": {"thread_id": session_id}},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue

        if "__interrupt__" in update:
            interrupts = update.get("__interrupt__") or []
            payloads = []
            for item in interrupts:
                payloads.append(getattr(item, "value", item))
            result["pending_approval"] = {
                "type": "plan_approval",
                "requests": payloads,
                "session_id": session_id,
            }
            progress = (
                "### 当前研究进度\n\n"
                "⏸ 研究计划正在等待人工确认。\n\n"
                f"可在CLI使用同一会话ID `{session_id}` 恢复。"
            )
            yield "__interrupt__", dict(result), progress
            return

        for current_node, node_update in update.items():
            if current_node not in NODE_NAMES:
                continue
            if isinstance(node_update, dict):
                result.update(node_update)

            event_round = int(result.get("round", 0) or 0)
            completed_lines.append(
                _completed_progress_line(current_node, event_round, result)
            )
            progress = _progress_markdown(completed_lines, current_node, result)
            yield current_node, dict(result), progress


def _score(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _cited_evidence_ids(result: dict[str, Any]) -> set[int]:
    final_answer = result.get("final_answer", "")
    if not isinstance(final_answer, str):
        return set()
    return {
        int(match)
        for match in re.findall(r"\[E([1-9]\d*)\]", final_answer)
    }


def build_evidence_table(result: dict[str, Any]) -> list[list[Any]]:
    """将已选证据转换为紧凑表格，不向页面暴露 Evidence 正文。"""
    rows = []
    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        return rows
    cited_ids = _cited_evidence_ids(result)

    for index, item in enumerate(evidence, 1):
        if not isinstance(item, dict):
            continue
        display_title = item.get("title") or item.get("source_name", "")
        content_type = "正文" if item.get("is_full_text") else "摘要"
        if item.get("source_type") == "local_rag" and item.get("parent_id"):
            content_type = (
                "父块上下文（含表格）"
                if "table" in str(item.get("content_types") or "")
                else "父块上下文"
            )
            trace = []
            page_start = item.get("page_start")
            page_end = item.get("page_end")
            if page_start:
                page_label = f"第{page_start}页"
                if page_end and page_end != page_start:
                    page_label = f"第{page_start}-{page_end}页"
                trace.append(page_label)
            if item.get("section_title"):
                trace.append(str(item["section_title"]))
            if item.get("ocr_used"):
                trace.append("OCR")
            if trace:
                display_title = f"{display_title}｜{' · '.join(trace)}"
            if item.get("parser_warnings"):
                content_type += "；⚠解析告警"
        rows.append([
            f"E{index}",
            SOURCE_LABELS.get(
                item.get("source_type", "unknown"),
                item.get("source_type", "未知"),
            ),
            display_title,
            _score(item.get("overall_score")),
            _score(item.get("relevance")),
            _score(item.get("semantic_relevance")),
            _score(item.get("authority")),
            content_type,
            "✓" if index in cited_ids else "—",
            item.get("url", ""),
        ])
    return rows


def _source_stats(result: dict[str, Any]) -> list[list[Any]]:
    retrieved = _source_counts(result, "all_evidence")
    selected = _source_counts(result, "evidence")
    return [
        [SOURCE_LABELS[source], retrieved[source], selected[source]]
        for source in SOURCE_ORDER
    ]


def _evidence_summary(result: dict[str, Any]) -> str:
    evidence = result.get("evidence", [])
    selected_count = len(evidence) if isinstance(evidence, list) else 0
    cited_count = len(
        _cited_evidence_ids(result).intersection(range(1, selected_count + 1))
    )
    return f"本轮筛选{selected_count}条候选证据，其中最终引用{cited_count}条。"


def _markdown_value(value: Any, empty_text: str = "暂无") -> str:
    if value is None or value == "":
        return empty_text
    if isinstance(value, str):
        return value
    return f"```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```"


def render_trace(record: dict[str, Any] | None) -> tuple[Any, ...]:
    """统一把一轮 research record 映射为右侧全部 Trace 组件。"""
    if not record:
        return (
            "### 当前研究进度\n\n等待提交研究问题。",
            "尚未进行研究。",
            {},
            "**研究轮次：** 0  \n**证据是否充分：** —",
            [],
            "本轮筛选0条候选证据，其中最终引用0条。",
            [],
            "暂无",
            [],
            "暂无",
            [],
            [],
            [],
            [],
        )

    result = record.get("result", {})
    if not isinstance(result, dict):
        result = {}
    question_context = (
        "### 原始问题\n\n"
        f"{record.get('user_question', '')}\n\n"
        "### 实际研究问题\n\n"
        f"{record.get('standalone_question', '')}"
    )
    intent_labels = {
        "clarify_previous": "解释上一轮结果",
        "deepen_previous": "深化上一轮研究",
        "new_topic": "新的研究主题",
    }
    intent = record.get("intent")
    if intent in intent_labels:
        question_context += f"\n\n**本轮类型：** {intent_labels[intent]}"
    sufficient = (
        "是" if result.get("sufficient") is True
        else "否" if result.get("sufficient") is False
        else "—"
    )
    workflow = (
        f"**研究轮次：** {result.get('round', 0)}  \n"
        f"**证据是否充分：** {sufficient}  \n"
        f"**长期记忆召回：** {len(result.get('recalled_memories', []) or [])}条"
        f"（{result.get('memory_recall_mode', 'disabled')}）"
    )
    progress = record.get("progress")
    if not progress:
        progress = (
            "### 当前研究进度\n\n✓ 研究完成"
            if result.get("final_answer")
            else "### 当前研究进度\n\n正在准备研究。"
        )

    return (
        progress,
        question_context,
        result.get("plan", {}) or {},
        workflow,
        _source_stats(result),
        _evidence_summary(result),
        build_evidence_table(result),
        _markdown_value(result.get("draft_answer")),
        _researcher_claims(result),
        _markdown_value(result.get("critique")),
        result.get("claim_reviews", []) or [],
        result.get("unsupported_claims", []) or [],
        result.get("missing_perspectives", []) or [],
        result.get("verified_claims", []) or [],
    )


def render_partial_trace(
    user_question: str,
    standalone_question: str,
    result: dict[str, Any],
    progress: str,
) -> tuple[Any, ...]:
    """复用统一渲染逻辑展示尚未结束的当前研究轮次。"""
    return render_trace({
        "user_question": user_question,
        "standalone_question": standalone_question,
        "result": result,
        "progress": progress,
    })


def _turn_label(index: int) -> str:
    return f"第{index}轮"


def _turn_index(selection: str | None) -> int | None:
    if not selection or not selection.startswith("第") or not selection.endswith("轮"):
        return None
    try:
        return int(selection[1:-1]) - 1
    except ValueError:
        return None


def select_trace(selection: str | None, records: list[dict[str, Any]]):
    """切换历史轮次时只更新右侧，不改聊天记录。"""
    index = _turn_index(selection)
    if index is None or not 0 <= index < len(records):
        return render_trace(None)
    return render_trace(records[index])


def clear_conversation():
    """清空当前浏览器会话，不触碰单例模型和 Graph。"""
    return (
        [],
        "",
        [],
        [],
        gr.Dropdown(choices=[], value=None, interactive=True),
        *render_trace(None),
    )


def make_message_handler(
    llm: QwenLLM,
    graph: Any,
    backend_lock: Lock,
    memory_store: ConversationMemoryStore,
):
    def handle_message(
        question: str,
        history: list[dict[str, Any]] | None,
        records: list[dict[str, Any]] | None,
        session_id: str,
        user_id: str,
        request: gr.Request,
    ):
        """随 LangGraph 节点进度同步更新对话与研究过程。"""
        question = (question or "").strip()
        history = list(history or [])
        records = list(records or [])
        session_id = (session_id or "").strip() or request.session_hash
        user_id = (user_id or "").strip()
        if not records and session_id:
            records = memory_store.load_session(session_id)
            if records and not history:
                for item in records:
                    history.extend([
                        {"role": "user", "content": item.get("user_question", "")},
                        {
                            "role": "assistant",
                            "content": item.get("result", {}).get("final_answer", ""),
                        },
                    ])
        history = history[-(MAX_UI_TURNS * 2):]
        records = records[-MAX_UI_TURNS:]

        if not question:
            choices = [_turn_label(i) for i in range(1, len(records) + 1)]
            selected = choices[-1] if choices else None
            trace = render_trace(records[-1] if records else None)
            yield (
                history,
                "",
                history,
                records,
                gr.Dropdown(choices=choices, value=selected, interactive=True),
                *trace,
            )
            return

        previous_history = list(history)
        turn_number = max(
            [int(item.get("turn", 0) or 0) for item in records if isinstance(item, dict)]
            or [0]
        ) + 1
        pending_label = _turn_label(turn_number)
        pending_choices = [
            _turn_label(i) for i in range(1, turn_number + 1)
        ]
        history.extend([
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "正在分析研究问题……",
            },
        ])
        standalone_question = question
        intent = "new_topic"
        intent_reason = ""
        topic_anchors = extract_topic_anchors(question)
        result: dict[str, Any] = {"question": question}
        progress = "### 当前研究进度\n\n● 正在分析研究问题……"
        trace = render_partial_trace(
            question,
            standalone_question,
            result,
            progress,
        )
        yield (
            history,
            "",
            history,
            records,
            gr.Dropdown(
                choices=pending_choices,
                value=pending_label,
                interactive=False,
            ),
            *trace,
        )

        try:
            # Qwen 与 Graph 共享模型实例；串行访问避免不同浏览器请求争用模型。
            with backend_lock:
                previous_record = records[-1] if records else None
                intent_result = classify_followup_intent(
                    llm,
                    question,
                    previous_record,
                    previous_history,
                )
                intent = intent_result["intent"]
                intent_reason = intent_result.get("reason", "")

                if intent == "clarify_previous" and previous_record:
                    topic_anchors = previous_record.get("topic_anchors", []) or (
                        extract_topic_anchors(
                            previous_record.get("standalone_question", "")
                        )
                    )
                    standalone_question = (
                        f"基于上一轮关于“{previous_record.get('standalone_question', '')}”"
                        f"的研究结果解释：{question}"
                    )
                    progress = (
                        "### 当前研究进度\n\n"
                        "● 正在结合上一轮研究结果解释……\n\n"
                        "本轮不会重新执行外部检索。"
                    )
                    history[-1] = {
                        "role": "assistant",
                        "content": "正在结合上一轮研究结果解释……",
                    }
                    yield (
                        history,
                        "",
                        history,
                        records,
                        gr.Dropdown(
                            choices=pending_choices,
                            value=pending_label,
                            interactive=False,
                        ),
                        *render_partial_trace(
                            question,
                            standalone_question,
                            {"question": standalone_question},
                            progress,
                        ),
                    )

                    final_answer = answer_clarification_from_record(
                        llm,
                        previous_record,
                        question,
                    )
                    previous_result = previous_record.get("result", {})
                    if not isinstance(previous_result, dict):
                        previous_result = {}
                    result = {
                        "question": standalone_question,
                        "evidence": previous_result.get("evidence", []) or [],
                        "verified_claims": previous_result.get(
                            "verified_claims", []
                        ) or [],
                        "unsupported_claims": previous_result.get(
                            "unsupported_claims", []
                        ) or [],
                        "missing_perspectives": previous_result.get(
                            "missing_perspectives", []
                        ) or [],
                        "sufficient": previous_result.get("sufficient"),
                        "round": 0,
                        "final_answer": final_answer,
                        "interaction_type": "clarification",
                    }
                    progress = (
                        "### 当前研究进度\n\n"
                        "✓ 已基于上一轮研究结果完成解释\n\n"
                        "本轮未重新执行外部检索。"
                    )
                else:
                    rewrite_record = (
                        previous_record if intent == "deepen_previous" else None
                    )
                    standalone_question, topic_anchors = rewrite_followup(
                        llm,
                        question,
                        previous_history,
                        previous_record=rewrite_record,
                    )
                    result = {"question": standalone_question}
                    history[-1] = {
                        "role": "assistant",
                        "content": "正在分析研究问题……",
                    }
                    yield (
                        history,
                        "",
                        history,
                        records,
                        gr.Dropdown(
                            choices=pending_choices,
                            value=pending_label,
                            interactive=False,
                        ),
                        *render_partial_trace(
                            question,
                            standalone_question,
                            result,
                            progress,
                        ),
                    )

                    received_update = False
                    for current_node, partial_result, progress in run_research_stream(
                        graph,
                        standalone_question,
                        session_id,
                        user_id,
                    ):
                        received_update = True
                        result = partial_result
                        history[-1] = {
                            "role": "assistant",
                            "content": _chat_status(current_node, result),
                        }
                        yield (
                            history,
                            "",
                            history,
                            records,
                            gr.Dropdown(
                                choices=pending_choices,
                                value=pending_label,
                                interactive=False,
                            ),
                            *render_partial_trace(
                                question,
                                standalone_question,
                                result,
                                progress,
                            ),
                        )

                    if not received_update or not result.get("final_answer"):
                        if result.get("pending_approval"):
                            final_answer = (
                                "研究计划正在等待人工确认。请使用页面中的会话 ID，"
                                "在 CLI 执行 --resume approve 或 --resume reject 后继续。"
                            )
                        else:
                            raise RuntimeError("LangGraph stream ended without a final answer")
                    else:
                        final_answer = result["final_answer"]
        except Exception:
            logger.exception("Research turn failed")
            history[-1] = {
                "role": "assistant",
                "content": "研究失败：本轮未能完成，请查看服务器日志后重试。",
            }
            choices = [_turn_label(i) for i in range(1, len(records) + 1)]
            selected = choices[-1] if choices else None
            error_progress = (
                "### 当前研究进度\n\n"
                "✗ 本轮研究未能完成，请查看服务器日志后重试。"
            )
            trace = render_partial_trace(
                question,
                standalone_question,
                result,
                error_progress,
            )
            yield (
                history,
                "",
                history,
                records,
                gr.Dropdown(
                    choices=choices,
                    value=selected,
                    interactive=True,
                ),
                *trace,
            )
            return

        history[-1] = {"role": "assistant", "content": str(final_answer)}
        record = {
            "turn": turn_number,
            "user_question": question,
            "standalone_question": standalone_question,
            "result": result,
            "progress": progress,
            "intent": intent,
            "intent_reason": intent_reason,
            "topic_anchors": topic_anchors,
        }
        records.append(record)
        history = history[-(MAX_UI_TURNS * 2):]
        records = records[-MAX_UI_TURNS:]
        memory_store.save_turn(
            session_id,
            turn_number,
            question,
            str(final_answer),
            record,
        )
        choices = [_turn_label(i) for i in range(1, len(records) + 1)]
        selected = choices[-1]
        yield (
            history,
            "",
            history,
            records,
            gr.Dropdown(choices=choices, value=selected, interactive=True),
            *render_trace(record),
        )

    return handle_message


def build_demo(
    llm: QwenLLM,
    graph: Any,
    memory_store: ConversationMemoryStore,
) -> gr.Blocks:
    """构建 Gradio 多轮对话页面。"""
    backend_lock = Lock()
    handle_message = make_message_handler(llm, graph, backend_lock, memory_store)

    with gr.Blocks(title="多源证据驱动研究智能体") as demo:
        gr.Markdown(
            "# 多源证据驱动研究智能体\n"
            "Web · GitHub · 学术论文 · 本地知识库 · 研究员 · 审查员"
        )

        chat_state = gr.State([])
        research_records = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="多轮研究对话",
                    height=650,
                )
                gr.Markdown(
                    "适合：需要论文、开源实现、近期资料和多方证据共同判断的问题。"
                )
                message = gr.Textbox(
                    label="问题",
                    placeholder="请输入研究问题；后续可以直接追问……",
                    lines=3,
                )
                session_id = gr.Textbox(
                    label="会话 ID（留空则使用当前浏览器会话；跨会话继续时填写原 ID）",
                    placeholder="例如 interview-prep-001",
                    lines=1,
                )
                user_id = gr.Textbox(
                    label="用户 ID（可选；复用同一 ID 可跨会话召回长期记忆）",
                    placeholder="例如 user-001；不填写则不启用长期记忆",
                    lines=1,
                )
                with gr.Row():
                    send_button = gr.Button("发送", variant="primary")
                    clear_button = gr.Button("清空对话")

                gr.Examples(
                    examples=[
                        ["让大语言模型 Agent 进行自我反思（Reflection）是否真的能够提高复杂推理能力？请结合论文、开源实现和近期资料分析其有效场景、局限及当前证据充分性。"],
                        ["如果现在要开发一个需要复杂状态管理、人工介入和长期维护的生产级 Agent，LangGraph、AutoGen 和 CrewAI 应该如何选择？请结合官方资料、GitHub 维护状态和工程特性分析。"],
                        ["RAG 系统中什么时候值得加入 reranker？请结合论文、开源实现和工程实践分析效果收益、延迟和计算成本。"],
                    ],
                    inputs=message,
                    label="示例问题",
                )

            with gr.Column(scale=2):
                gr.Markdown("## 研究过程")
                current_progress = gr.Markdown(
                    "### 当前研究进度\n\n等待提交研究问题。"
                )
                turn_selector = gr.Dropdown(
                    label="查看研究轮次",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                question_context = gr.Markdown("尚未进行研究。")

                with gr.Accordion("研究规划", open=True):
                    planner = gr.JSON(label="研究计划", value={})

                with gr.Accordion("工作流状态", open=True):
                    workflow = gr.Markdown(
                        "**研究轮次：** 0  \n**证据是否充分：** —"
                    )

                with gr.Accordion("检索来源统计", open=True):
                    source_stats = gr.Dataframe(
                        headers=["来源", "检索数量", "筛选数量"],
                        datatype=["str", "number", "number"],
                        value=[],
                        interactive=False,
                    )

                with gr.Accordion("候选证据", open=True):
                    evidence_summary = gr.Markdown(
                        "本轮筛选0条候选证据，其中最终引用0条。"
                    )
                    evidence_table = gr.Dataframe(
                        headers=EVIDENCE_HEADERS,
                        datatype=[
                            "str", "str", "str", "number", "number",
                            "number", "number", "str", "str", "str",
                        ],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )

                with gr.Accordion("研究员分析", open=False):
                    draft = gr.Markdown("暂无")
                    claims = gr.JSON(label="候选主张", value=[])

                with gr.Accordion("审查员评审", open=False):
                    critique = gr.Markdown("暂无")
                    claim_reviews = gr.JSON(
                        label="逐项主张审查",
                        value=[],
                    )
                    unsupported_claims = gr.JSON(
                        label="证据不足的主张",
                        value=[],
                    )
                    missing_perspectives = gr.JSON(
                        label="缺失视角",
                        value=[],
                    )

                with gr.Accordion("已验证主张", open=False):
                    verified_claims = gr.JSON(label="已验证主张", value=[])

        trace_outputs = [
            current_progress,
            question_context,
            planner,
            workflow,
            source_stats,
            evidence_summary,
            evidence_table,
            draft,
            claims,
            critique,
            claim_reviews,
            unsupported_claims,
            missing_perspectives,
            verified_claims,
        ]
        message_outputs = [
            chatbot,
            message,
            chat_state,
            research_records,
            turn_selector,
            *trace_outputs,
        ]

        send_button.click(
            fn=handle_message,
            inputs=[message, chat_state, research_records, session_id, user_id],
            outputs=message_outputs,
            concurrency_limit=1,
            concurrency_id="research_backend",
        )
        message.submit(
            fn=handle_message,
            inputs=[message, chat_state, research_records, session_id, user_id],
            outputs=message_outputs,
            concurrency_limit=1,
            concurrency_id="research_backend",
        )
        turn_selector.input(
            fn=select_trace,
            inputs=[turn_selector, research_records],
            outputs=trace_outputs,
        )
        clear_button.click(
            fn=clear_conversation,
            inputs=None,
            outputs=message_outputs,
            concurrency_limit=1,
            concurrency_id="research_backend",
        )

    return demo


def main():
    configure_logging(logging.INFO)
    llm, graph, memory_store = build_research_backend()
    demo = build_demo(llm, graph, memory_store)
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
