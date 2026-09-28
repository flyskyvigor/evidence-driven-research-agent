"""模型工具选择与确定性回退。

优先使用模型原生工具模板；若模型或模板不支持，则根据已验证 Planner 结果构造
同样的 ToolCall。回退会明确记录，不能把它冒充为模型原生调用。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from research_agent.tools.models import ToolCall
from research_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolSelector:
    def __init__(self, llm: Any, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry
        self.last_error = ""

    def select(self, question: str, plan: dict[str, Any]) -> tuple[list[ToolCall], str]:
        self.last_error = ""
        selectable = {
            "web.web_search",
            "github.get_repository",
            "github.search_repositories",
            "paper.search_papers",
            "local.search_knowledge",
        }
        prompt = (
            "你是研究工具路由器。根据问题和已验证研究计划选择必要工具。"
            "只调用与计划一致的工具；可以为不同查询多次调用同一工具。"
            "不要抓取尚未由 web_search 返回的 URL。\n\n"
            f"问题：{question}\n计划：{json.dumps(plan, ensure_ascii=False)}"
        )
        try:
            calls = self.llm.generate_tool_calls(
                prompt,
                [
                    spec.as_function_schema()
                    for spec in self.registry.specs()
                    if spec.name in selectable
                ],
                max_calls=14,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("model_tool_selection_fallback error=%s", self.last_error)
            calls = []
        allowed = set(self.registry.names()) & selectable
        normalized_calls = []
        for call in calls:
            normalized_name = call.name
            if normalized_name not in allowed:
                candidate = normalized_name.replace("__", ".")
                if candidate in allowed:
                    normalized_name = candidate
            if normalized_name in allowed:
                normalized_calls.append(ToolCall(
                    name=normalized_name,
                    arguments=call.arguments,
                    call_id=call.call_id,
                ))
        calls = normalized_calls
        if calls:
            self.last_error = ""
            return calls, "model_native"
        if not self.last_error:
            self.last_error = "model_returned_no_valid_tool_calls"
        return self._fallback(plan), "deterministic_fallback"

    @staticmethod
    def _fallback(plan: dict[str, Any]) -> list[ToolCall]:
        calls = []
        for query in plan.get("queries", [])[:4]:
            calls.append(ToolCall("web.web_search", {"query": query, "max_results": 4}))
        for repo in plan.get("github_repos", [])[:3]:
            calls.append(ToolCall("github.get_repository", {"full_name": repo}))
        for query in plan.get("github_queries", [])[:2]:
            calls.append(ToolCall("github.search_repositories", {"query": query, "max_results": 1}))
        for query in plan.get("paper_queries", [])[:3]:
            calls.append(ToolCall("paper.search_papers", {"query": query, "max_results": 4}))
        goal = str(plan.get("goal") or "").strip()
        if goal:
            calls.append(ToolCall("local.search_knowledge", {"query": goal, "top_k": 4}))
        return calls

