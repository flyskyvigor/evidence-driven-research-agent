import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from research_agent.config import (
    get_semantic_scholar_api_key,
    get_web_proxy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "mcp_servers.paper_server"


class MCPPaperRetriever:
    def search(self, query, max_results=4):
        result = self._run(
            "search_papers",
            {"query": query, "max_results": max_results}
        )

        if isinstance(result, dict):
            result = (
                result.get("papers")
                or result.get("results")
                or []
            )

        return result if isinstance(result, list) else []

    def get_paper(self, paper_id):
        result = self._run(
            "get_paper",
            {"paper_id": paper_id}
        )
        return result if isinstance(result, dict) else {}

    def _run(self, tool_name, arguments):
        try:
            return asyncio.run(self._call_tool(tool_name, arguments))
        except Exception:
            return [] if tool_name == "search_papers" else {}

    async def _call_tool(self, tool_name, arguments):
        env = os.environ.copy()
        proxy = get_web_proxy()
        if proxy:
            env["WEB_PROXY"] = proxy
        else:
            env.pop("WEB_PROXY", None)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH")])
        )

        api_key = get_semantic_scholar_api_key()
        if api_key:
            env["SEMANTIC_SCHOLAR_API_KEY"] = api_key

        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULE],
            env=env
        )

        async with Client(stdio_client(server)) as client:
            result = await client.call_tool(tool_name, arguments)

        if result.is_error:
            return None

        if result.structured_content:
            return self._unwrap(result.structured_content)

        for item in result.content:
            if isinstance(item, TextContent):
                try:
                    return self._unwrap(json.loads(item.text))
                except json.JSONDecodeError:
                    continue

        return None

    @classmethod
    def _unwrap(cls, data):
        if isinstance(data, dict) and "result" in data:
            return cls._unwrap(data["result"])
        return data
