"""GitHub MCP 工具的领域适配器。"""

from __future__ import annotations

from research_agent.tools.runtime import ToolRuntime, build_default_tool_runtime


class MCPGitHubRetriever:
    def __init__(self, runtime: ToolRuntime | None = None) -> None:
        self.runtime = runtime or build_default_tool_runtime()
        self.last_result = None

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        self.last_result = self.runtime.execute(
            "github.search_repositories",
            {"query": query, "max_results": max_results},
        )
        return self.last_result.data if self.last_result.success and isinstance(self.last_result.data, list) else []

    def get_repository(self, full_name: str) -> dict:
        self.last_result = self.runtime.execute(
            "github.get_repository",
            {"full_name": full_name},
        )
        return self.last_result.data if self.last_result.success and isinstance(self.last_result.data, dict) else {}
