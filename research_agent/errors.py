"""项目级异常分类。

异常类型是工具可靠性和评测的基础：调用方必须能区分参数错误、超时、
外部服务故障、协议故障和证据不足，而不是把所有错误都转换为空列表。
"""

from __future__ import annotations


class ResearchAgentError(Exception):
    """所有可预期项目异常的基类。"""


class ConfigurationError(ResearchAgentError):
    """配置缺失或配置值非法。"""


class ToolError(ResearchAgentError):
    """工具错误基类。"""

    retryable = False


class ToolNotFoundError(ToolError):
    """请求了未注册工具。"""


class ToolValidationError(ToolError):
    """工具参数没有通过 JSON Schema 子集校验。"""


class ToolTimeoutError(ToolError):
    """工具调用超过总超时。"""

    retryable = True


class ToolProtocolError(ToolError):
    """MCP 握手、发现、调用或返回格式不符合预期。"""


class ExternalServiceError(ToolError):
    """远端搜索/API/页面服务临时失败。"""

    retryable = True


class ModelOutputError(ResearchAgentError):
    """模型输出无法解析或不满足结构约束。"""


class EvidenceValidationError(ResearchAgentError):
    """证据或主张结构不满足约束。"""


class DocumentParseError(ResearchAgentError):
    """本地文档无法解析，或请求的解析/OCR能力不可用。"""

