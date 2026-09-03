import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from research_agent.config import get_web_proxy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "mcp_servers.web_server"


class MCPWebRetriever:
    def search(self, query, max_results=4):
        result = self._run("web_search", {"query": query, "max_results": max_results})
        return result if isinstance(result, list) else []

    def fetch(self, url):
        result = self._run("fetch_page", {"url": url})
        return result if isinstance(result, dict) else {}

    def _run(self, tool_name, arguments):
        try:
            return asyncio.run(self._call_tool(tool_name, arguments))
        except Exception:
            return [] if tool_name == "web_search" else {}

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
        env["RUST_LOG"] = "error"

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
