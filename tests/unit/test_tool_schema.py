import pytest

from research_agent.errors import ToolValidationError
from research_agent.tools.schema import validate_arguments


SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def test_valid_arguments_are_returned():
    assert validate_arguments({"query": "agent", "limit": 2}, SCHEMA)["limit"] == 2


@pytest.mark.parametrize(
    "arguments",
    [{}, {"query": "x"}, {"query": "agent", "limit": 9}, {"query": "agent", "extra": 1}],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(ToolValidationError):
        validate_arguments(arguments, SCHEMA)
