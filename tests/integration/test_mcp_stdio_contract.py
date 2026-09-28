import sys

import pytest

from research_agent.mcp.client import MCPGateway, MCPServerConfig


@pytest.mark.integration
def test_stdio_tool_discovery_and_call_without_public_network():
    gateway = MCPGateway([
        MCPServerConfig(
            name="echo",
            command=sys.executable,
            args=("-m", "tests.fixtures.echo_mcp_server"),
            timeout_seconds=10,
            max_retries=0,
        )
    ])
    tools = gateway.discover_server("echo")
    assert {item.remote_name for item in tools} == {"echo"}
    assert gateway.call("echo", "echo", {"text": "a", "repeat": 2}) == {"text": "aa"}
