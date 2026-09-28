"""线程安全工具注册表。"""

from __future__ import annotations

from threading import RLock

from research_agent.errors import ToolNotFoundError
from research_agent.tools.models import RegisteredTool, ToolHandler, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._lock = RLock()

    def register(self, spec: ToolSpec, handler: ToolHandler, *, replace: bool = False) -> None:
        with self._lock:
            if spec.name in self._tools and not replace:
                raise ValueError(f"Tool already registered: {spec.name}")
            self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get(self, name: str) -> RegisteredTool:
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as exc:
                raise ToolNotFoundError(f"Unknown tool: {name}") from exc

    def specs(self) -> list[ToolSpec]:
        with self._lock:
            return [item.spec for item in self._tools.values()]

    def function_schemas(self) -> list[dict]:
        return [spec.as_function_schema() for spec in self.specs()]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

