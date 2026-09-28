"""Web MCP 工具的领域适配器。"""

from __future__ import annotations

from research_agent.tools.runtime import ToolRuntime, build_default_tool_runtime


class MCPWebRetriever:
    def __init__(self, runtime: ToolRuntime | None = None) -> None:
        self.runtime = runtime or build_default_tool_runtime()
        self.last_result = None

    def search(self, query: str, max_results: int = 4) -> list[dict]:
        self.last_result = self.runtime.execute(
            "web.web_search",
            {"query": query, "max_results": max_results},
        )
        return self.last_result.data if self.last_result.success and isinstance(self.last_result.data, list) else []

    def fetch(self, url: str) -> dict:
        self.last_result = self.runtime.execute("web.fetch_page", {"url": url})
        return self.last_result.data if self.last_result.success and isinstance(self.last_result.data, dict) else {}
