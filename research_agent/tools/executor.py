"""带校验、总超时、有限重试和标准结果的工具执行器。"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from research_agent.errors import (
    ExternalServiceError,
    ToolError,
    ToolNotFoundError,
    ToolProtocolError,
    ToolTimeoutError,
    ToolValidationError,
)
from research_agent.tools.models import ToolCall, ToolFailure, ToolResult, ToolStatus
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.schema import validate_arguments


logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 8) -> None:
        self.registry = registry
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            registered = self.registry.get(call.name)
            validate_arguments(call.arguments, registered.spec.input_schema)
        except ToolNotFoundError as exc:
            return self._failure(call, ToolStatus.NOT_FOUND, exc, started, 0)
        except ToolValidationError as exc:
            return self._failure(call, ToolStatus.INVALID_ARGUMENTS, exc, started, 0)

        attempts = max(1, registered.spec.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            future = self._pool.submit(registered.handler, dict(call.arguments))
            try:
                data = future.result(timeout=registered.spec.timeout_seconds)
                data = self._limit_result(data, registered.spec.result_limit_chars)
                duration_ms = (time.perf_counter() - started) * 1000
                logger.info(
                    "tool_call_completed tool=%s status=success attempts=%s duration_ms=%.1f",
                    call.name,
                    attempt,
                    duration_ms,
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    status=ToolStatus.SUCCESS,
                    data=data,
                    duration_ms=duration_ms,
                    attempts=attempt,
                    metadata={"source": registered.spec.source},
                )
            except FutureTimeout:
                future.cancel()
                last_error = ToolTimeoutError(
                    f"Tool {call.name} exceeded {registered.spec.timeout_seconds:.1f}s."
                )
            except Exception as exc:  # handler boundary: normalize every failure
                last_error = exc

            retryable = bool(getattr(last_error, "retryable", False))
            if attempt >= attempts or not retryable:
                break
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))

        assert last_error is not None
        status = self._status_for(last_error)
        result = self._failure(call, status, last_error, started, attempt)
        logger.warning(
            "tool_call_completed tool=%s status=%s attempts=%s duration_ms=%.1f error_type=%s",
            call.name,
            status.value,
            attempt,
            result.duration_ms,
            result.failure.error_type if result.failure else "",
        )
        return result

    @staticmethod
    def _limit_result(data: Any, limit: int) -> Any:
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        if len(encoded) <= limit:
            return data
        if isinstance(data, list):
            kept = []
            size = 2
            for item in data:
                item_size = len(json.dumps(item, ensure_ascii=False, default=str)) + 1
                if size + item_size > limit:
                    break
                kept.append(item)
                size += item_size
            return kept
        if isinstance(data, dict):
            return {"truncated": True, "preview": encoded[:limit]}
        return str(data)[:limit]

    @staticmethod
    def _status_for(exc: Exception) -> ToolStatus:
        if isinstance(exc, ToolValidationError):
            return ToolStatus.INVALID_ARGUMENTS
        if isinstance(exc, ToolNotFoundError):
            return ToolStatus.NOT_FOUND
        if isinstance(exc, ToolTimeoutError):
            return ToolStatus.TIMEOUT
        if isinstance(exc, ToolProtocolError):
            return ToolStatus.PROTOCOL_ERROR
        if isinstance(exc, ExternalServiceError):
            return ToolStatus.EXTERNAL_ERROR
        return ToolStatus.FAILED

    @staticmethod
    def _failure(
        call: ToolCall,
        status: ToolStatus,
        exc: Exception,
        started: float,
        attempts: int,
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            status=status,
            failure=ToolFailure(
                error_type=type(exc).__name__,
                message=str(exc)[:500],
                retryable=bool(getattr(exc, "retryable", False)),
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
            attempts=max(1, attempts),
        )

