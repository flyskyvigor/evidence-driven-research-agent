from research_agent.errors import ExternalServiceError
from research_agent.tools.executor import ToolExecutor
from research_agent.tools.models import ToolCall, ToolSpec, ToolStatus
from research_agent.tools.registry import ToolRegistry


def _spec(retries=0):
    return ToolSpec(
        name="test.echo",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        max_retries=retries,
    )


def test_executor_normalizes_success():
    registry = ToolRegistry()
    registry.register(_spec(), lambda args: {"text": args["text"]})
    result = ToolExecutor(registry).execute(ToolCall("test.echo", {"text": "ok"}))
    assert result.status == ToolStatus.SUCCESS
    assert result.data == {"text": "ok"}


def test_executor_retries_retryable_failure():
    attempts = {"count": 0}

    def flaky(args):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ExternalServiceError("temporary")
        return args

    registry = ToolRegistry()
    registry.register(_spec(retries=1), flaky)
    result = ToolExecutor(registry).execute(ToolCall("test.echo", {"text": "ok"}))
    assert result.success
    assert result.attempts == 2
