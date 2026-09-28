"""Paper MCP 工具的领域适配器。"""

from __future__ import annotations

from research_agent.tools.runtime import ToolRuntime, build_default_tool_runtime


class MCPPaperRetriever:
    def __init__(self, runtime: ToolRuntime | None = None) -> None:
        self.runtime = runtime or build_default_tool_runtime()
        self.last_result = None

    def search(self, query: str, max_results: int = 4) -> list[dict]:
        self.last_result = self.runtime.execute(
            "paper.search_papers",
            {"query": query, "max_results": max_results},
        )
        data = self.last_result.data if self.last_result.success else []
        if isinstance(data, dict):
            data = data.get("papers") or data.get("results") or []
        return data if isinstance(data, list) else []

    def get_paper(self, paper_id: str) -> dict:
        self.last_result = self.runtime.execute("paper.get_paper", {"paper_id": paper_id})
        return self.last_result.data if self.last_result.success and isinstance(self.last_result.data, dict) else {}
