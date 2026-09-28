"""轻量 JSON Schema 参数校验。

项目只实现工具参数所需的稳定子集，避免把“有 Schema”误写成“已校验”。
复杂 Schema 应改用专门验证库；这里遇到不支持的关键字会保守忽略并由工具端复核。
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from research_agent.errors import ToolValidationError


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def validate_arguments(arguments: Any, schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be a JSON object.")
    _validate(arguments, schema or {"type": "object"}, path="arguments")
    return arguments


def _validate(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_matches_type(value, item) for item in expected):
            raise ToolValidationError(f"{path} has an invalid type.")
    elif expected and not _matches_type(value, expected):
        raise ToolValidationError(f"{path} must be {expected}.")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of {schema['enum']}.")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ToolValidationError(f"{path}.{required} is required.")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ToolValidationError(f"{path} contains unknown fields: {unknown}.")
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate(item, child_schema, f"{path}.{key}")

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < int(minimum):
            raise ToolValidationError(f"{path} contains too few items.")
        if maximum is not None and len(value) > int(maximum):
            raise ToolValidationError(f"{path} contains too many items.")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        if schema.get("minLength") is not None and len(value) < int(schema["minLength"]):
            raise ToolValidationError(f"{path} is too short.")
        if schema.get("maxLength") is not None and len(value) > int(schema["maxLength"]):
            raise ToolValidationError(f"{path} is too long.")
        if schema.get("pattern") and not re.search(str(schema["pattern"]), value):
            raise ToolValidationError(f"{path} does not match the required pattern.")
        if schema.get("format") == "uri":
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ToolValidationError(f"{path} must be an HTTP(S) URI.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.get("minimum") is not None and value < schema["minimum"]:
            raise ToolValidationError(f"{path} is below the minimum.")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            raise ToolValidationError(f"{path} exceeds the maximum.")


def _matches_type(value: Any, expected: str) -> bool:
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return True
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)

