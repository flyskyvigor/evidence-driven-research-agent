"""统一工具运行时及内置 MCP 工具清单。"""

from __future__ import annotations

import os
import sys
from threading import Lock
from typing import Any

from research_agent.config import (
    get_github_token,
    get_semantic_scholar_api_key,
    get_web_proxy,
)
from research_agent.mcp.client import MCPGateway, MCPServerConfig
from research_agent.errors import ExternalServiceError
from research_agent.tools.executor import ToolExecutor
from research_agent.tools.models import ToolCall, ToolResult, ToolSpec
from research_agent.tools.registry import ToolRegistry


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, gateway: MCPGateway) -> None:
        self.registry = registry
        self.gateway = gateway
        self.executor = ToolExecutor(registry)
        self.discovery_failures: dict[str, str] = {}
        self._discovery_lock = Lock()
        self._discovery_attempted = False
        self._history_lock = Lock()
        self._history: list[ToolResult] = []

    def discover_mcp_tools(self, *, force: bool = False) -> list[ToolSpec]:
        with self._discovery_lock:
            if self._discovery_attempted and not force:
                return self.registry.specs()
            discovered, failures = self.gateway.discover_all()
            self.discovery_failures = failures
            for tool in discovered:
                config = self.gateway.servers[tool.server_name]
                spec = tool.to_spec(config)
                self.registry.register(
                    spec,
                    _mcp_handler(self.gateway, tool.server_name, tool.remote_name),
                    replace=True,
                )
            self._discovery_attempted = True
            return self.registry.specs()

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.execute_call(ToolCall(name=name, arguments=arguments))

    def execute_call(self, call: ToolCall) -> ToolResult:
        result = self.executor.execute(call)
        with self._history_lock:
            self._history.append(result)
        return result

    def history_since(self, index: int) -> list[ToolResult]:
        with self._history_lock:
            return list(self._history[index:])

    def history_size(self) -> int:
        with self._history_lock:
            return len(self._history)


def build_default_tool_runtime() -> ToolRuntime:
    proxy = get_web_proxy()
    shared_env = {"WEB_PROXY": proxy} if proxy else {}
    github_env = dict(shared_env)
    paper_env = dict(shared_env)
    if get_github_token():
        github_env["GITHUB_TOKEN"] = get_github_token() or ""
    if get_semantic_scholar_api_key():
        paper_env["SEMANTIC_SCHOLAR_API_KEY"] = get_semantic_scholar_api_key() or ""

    configs = [
        MCPServerConfig("web", sys.executable, ("-m", "mcp_servers.web_server"), shared_env, 35.0, 1),
        MCPServerConfig("github", sys.executable, ("-m", "mcp_servers.github_server"), github_env, 40.0, 1),
        MCPServerConfig("paper", sys.executable, ("-m", "mcp_servers.paper_server"), paper_env, 45.0, 1),
    ]
    gateway = MCPGateway(configs)
    registry = ToolRegistry()
    runtime = ToolRuntime(registry, gateway)
    _register_builtin_manifests(runtime)
    return runtime


def _mcp_handler(gateway: MCPGateway, server: str, tool: str):
    def handler(arguments):
        data = gateway.call(server, tool, arguments)
        if isinstance(data, dict) and data.get("error"):
            raise ExternalServiceError(
                f"{server}.{tool} returned {data.get('error')}"
            )
        return data

    return handler


def _register_builtin_manifests(runtime: ToolRuntime) -> None:
    """离线时仍可向模型展示 Schema；在线发现结果会覆盖这些清单。"""
    definitions = [
        ("web", "web_search", "搜索公开网页并返回候选证据。", {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 160},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
            },
            "required": ["query"],
            "additionalProperties": False,
        }),
        ("web", "fetch_page", "抓取公开 HTTP(S) 页面并提取正文。", {
            "type": "object",
            "properties": {"url": {"type": "string", "format": "uri", "maxLength": 2048}},
            "required": ["url"],
            "additionalProperties": False,
        }),
        ("github", "search_repositories", "搜索公开 GitHub 仓库。", {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 100},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        }),
        ("github", "get_repository", "读取准确的 owner/repository 公共仓库。", {
            "type": "object",
            "properties": {
                "full_name": {
                    "type": "string",
                    "pattern": r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$",
                    "maxLength": 140,
                }
            },
            "required": ["full_name"],
            "additionalProperties": False,
        }),
        ("paper", "search_papers", "从学术数据源搜索论文元数据和摘要。", {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 180},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
            },
            "required": ["query"],
            "additionalProperties": False,
        }),
        ("paper", "get_paper", "按论文标识读取详细元数据。", {
            "type": "object",
            "properties": {"paper_id": {"type": "string", "minLength": 1, "maxLength": 300}},
            "required": ["paper_id"],
            "additionalProperties": False,
        }),
    ]
    for server, remote, description, schema in definitions:
        config = runtime.gateway.servers[server]
        runtime.registry.register(
            ToolSpec(
                name=f"{server}.{remote}",
                description=description,
                input_schema=schema,
                source="mcp_manifest",
                server_name=server,
                remote_name=remote,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            ),
            _mcp_handler(runtime.gateway, server, remote),
        )

    runtime.registry.register(
        ToolSpec(
            name="local.search_knowledge",
            description=(
                "在本地论文/文档父子索引中执行稠密向量与BM25混合召回，"
                "使用RRF融合并返回带页码、章节和子块追溯的上下文父块。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "maxLength": 500},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 4},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            source="python",
            timeout_seconds=60.0,
            max_retries=0,
        ),
        _local_knowledge_handler,
    )


def _local_knowledge_handler(arguments):
    # 延迟导入避免 Tool Runtime 与 RAG 初始化形成循环依赖。
    from research_agent.rag.knowledge_base import retrieve_knowledge

    return retrieve_knowledge(
        arguments["query"],
        top_k=int(arguments.get("top_k", 4)),
    )

