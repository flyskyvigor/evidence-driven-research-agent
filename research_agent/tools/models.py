"""Tool Calling 的稳定领域模型。

这些对象把“模型选择工具”和“Python 执行函数”分开：模型只生成 ToolCall，
执行器负责验证、超时、重试和把任何返回值转换成 ToolResult。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolStatus(str, Enum):
    SUCCESS = "success"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    PROTOCOL_ERROR = "protocol_error"
    EXTERNAL_ERROR = "external_error"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "python"
    server_name: str | None = None
    remote_name: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    result_limit_chars: int = 120_000

    def as_function_schema(self) -> dict[str, Any]:
        """转换成 Transformers/OpenAI 风格的模型工具 Schema。"""
        return {
            "type": "function",
            "function": {
                # 常见工具模板只接受字母、数字、下划线和连字符。
                "name": self.name.replace(".", "__"),
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:16]}")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ToolCall":
        function = value.get("function") if isinstance(value.get("function"), dict) else value
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must be an object.")
        return cls(
            name=name,
            arguments=arguments,
            call_id=str(value.get("id") or value.get("call_id") or f"call_{uuid.uuid4().hex[:16]}"),
        )


@dataclass(frozen=True)
class ToolFailure:
    error_type: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolStatus
    data: Any = None
    failure: ToolFailure | None = None
    duration_ms: float = 0.0
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    def as_model_message(self) -> dict[str, Any]:
        """生成可追加到聊天模板的 tool role 消息。"""
        return {
            "role": "tool",
            "name": self.tool_name,
            "tool_call_id": self.call_id,
            "content": json.dumps(self.as_dict(), ensure_ascii=False, default=str),
        }


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler

