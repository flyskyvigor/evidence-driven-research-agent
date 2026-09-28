"""统一 MCP Client。

MCP 是模型上下文协议；具体 Web/GitHub/Paper HTTP API 只是 MCP Server 背后的
数据源。本模块负责 stdio 进程、工具发现、协议超时和结果解包，不包含业务检索逻辑。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Coroutine

from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from research_agent.errors import ExternalServiceError, ToolProtocolError, ToolTimeoutError
from research_agent.tools.models import ToolSpec


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str = sys.executable
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 35.0
    max_retries: int = 1
    enabled: bool = True

    def parameters(self) -> StdioServerParameters:
        env = os.environ.copy()
        env.update(self.env)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH")])
        )
        env.setdefault("RUST_LOG", "error")
        return StdioServerParameters(command=self.command, args=list(self.args), env=env)


@dataclass(frozen=True)
class DiscoveredMCPTool:
    server_name: str
    remote_name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{self.server_name}.{self.remote_name}"

    def to_spec(self, config: MCPServerConfig) -> ToolSpec:
        return ToolSpec(
            name=self.qualified_name,
            description=self.description or self.remote_name,
            input_schema=self.input_schema or {"type": "object"},
            source="mcp",
            server_name=self.server_name,
            remote_name=self.remote_name,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )


class MCPGateway:
    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = {item.name: item for item in servers if item.enabled}
        self._discovered: dict[str, DiscoveredMCPTool] = {}

    def discover_server(self, server_name: str) -> list[DiscoveredMCPTool]:
        config = self._server(server_name)
        try:
            tools = _run_async(
                self._discover_async(config),
                timeout=config.timeout_seconds,
            )
        except (asyncio.TimeoutError, ToolTimeoutError) as exc:
            raise ToolTimeoutError(f"MCP tool discovery timed out: {server_name}") from exc
        except Exception as exc:
            raise ToolProtocolError(
                f"MCP tool discovery failed for {server_name}: {type(exc).__name__}: {exc}"
            ) from exc

        for tool in tools:
            self._discovered[tool.qualified_name] = tool
        return tools

    def discover_all(self) -> tuple[list[DiscoveredMCPTool], dict[str, str]]:
        discovered = []
        failures = {}
        for server_name in self.servers:
            try:
                discovered.extend(self.discover_server(server_name))
            except Exception as exc:
                failures[server_name] = f"{type(exc).__name__}: {exc}"
                logger.warning("mcp_discovery_failed server=%s error=%s", server_name, exc)
        return discovered, failures

    def call(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        config = self._server(server_name)
        try:
            return _run_async(
                self._call_async(config, tool_name, arguments),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ToolTimeoutError(
                f"MCP call timed out: {server_name}.{tool_name}"
            ) from exc
        except (ExternalServiceError, ToolProtocolError, ToolTimeoutError):
            raise
        except Exception as exc:
            raise ToolProtocolError(
                f"MCP call failed for {server_name}.{tool_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def _discover_async(self, config: MCPServerConfig) -> list[DiscoveredMCPTool]:
        found = []
        cursor = None
        async with Client(config.parameters(), read_timeout_seconds=config.timeout_seconds) as client:
            while True:
                result = await client.list_tools(cursor=cursor)
                for item in result.tools:
                    schema = getattr(item, "input_schema", None)
                    if schema is None:
                        schema = getattr(item, "inputSchema", None)
                    found.append(DiscoveredMCPTool(
                        server_name=config.name,
                        remote_name=item.name,
                        description=getattr(item, "description", "") or "",
                        input_schema=schema or {"type": "object"},
                    ))
                cursor = getattr(result, "next_cursor", None)
                if cursor is None:
                    break
        return found

    async def _call_async(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        async with Client(config.parameters(), read_timeout_seconds=config.timeout_seconds) as client:
            result = await client.call_tool(tool_name, arguments)

        if result.is_error:
            detail = _text_content(result.content) or "MCP tool returned is_error=true."
            raise ExternalServiceError(detail[:500])
        if result.structured_content is not None:
            return _unwrap(result.structured_content)
        text = _text_content(result.content)
        if not text:
            return None
        try:
            return _unwrap(json.loads(text))
        except json.JSONDecodeError:
            return text

    def _server(self, name: str) -> MCPServerConfig:
        try:
            return self.servers[name]
        except KeyError as exc:
            raise ToolProtocolError(f"Unknown MCP server: {name}") from exc


def _text_content(content: list[Any]) -> str:
    parts = [item.text for item in content if isinstance(item, TextContent)]
    return "\n".join(parts).strip()


def _unwrap(data: Any) -> Any:
    while isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]
    return data


def _run_async(coro: Coroutine[Any, Any, Any], timeout: float) -> Any:
    """在同步 LangGraph 节点中安全执行异步 MCP 调用。"""

    async def guarded() -> Any:
        return await asyncio.wait_for(coro, timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(guarded())

    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = asyncio.run(guarded())
        except BaseException as exc:  # transfer exception across thread boundary
            outcome["error"] = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout + 1.0)
    if thread.is_alive():
        raise ToolTimeoutError("Async MCP runner did not stop after timeout.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")

