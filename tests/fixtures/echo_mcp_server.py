"""不访问公网的真实 stdio MCP 测试 Server。"""

from mcp.server import MCPServer


mcp = MCPServer("ReasoningAgent Echo Test Server")


@mcp.tool()
def echo(text: str, repeat: int = 1) -> dict:
    """Return text for protocol contract tests."""
    repeat = max(1, min(int(repeat), 3))
    return {"text": text * repeat}


if __name__ == "__main__":
    mcp.run(transport="stdio")
