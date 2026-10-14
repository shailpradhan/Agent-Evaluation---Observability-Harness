"""OpenTelemetry span definitions and context managers for agent execution."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry.trace import Span, StatusCode, Tracer

# Span Names
SPAN_AGENT_RUN = "agent.run"
SPAN_LLM_CALL = "llm.call"
SPAN_TOOL_CALL = "tool.call"

# Semantic Attribute Keys
ATTR_AGENT_RUN_ID = "agent.run_id"
ATTR_AGENT_TASK_ID = "agent.task_id"
ATTR_AGENT_MODEL = "agent.model"
ATTR_AGENT_FINAL_RESPONSE = "agent.final_response"

ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_PROMPT = "llm.prompt"
ATTR_LLM_RESPONSE = "llm.response"
ATTR_LLM_TOTAL_TOKENS = "llm.tokens.total"
ATTR_LLM_LATENCY_MS = "llm.latency_ms"

ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_INPUT = "tool.input"
ATTR_TOOL_OUTPUT = "tool.output"
ATTR_TOOL_ERROR = "tool.error"
ATTR_TOOL_LATENCY_MS = "tool.latency_ms"


def _safe_json_dumps(value: Any) -> str:
    """Serialize value to JSON string safely."""
    try:
        if hasattr(value, "model_dump_json"):
            return value.model_dump_json()
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump(mode="json"))
        return json.dumps(value, default=str)
    except (TypeError, ValueError, AttributeError):
        return str(value)


@contextmanager
def trace_agent_run(
    tracer: Tracer,
    run_id: str,
    task_id: str,
    model_name: str,
) -> Generator[Span, None, None]:
    """Span context manager for an overall agent execution run."""
    with tracer.start_as_current_span(
        SPAN_AGENT_RUN,
        attributes={
            ATTR_AGENT_RUN_ID: run_id,
            ATTR_AGENT_TASK_ID: task_id,
            ATTR_AGENT_MODEL: model_name,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


@contextmanager
def trace_llm_call(
    tracer: Tracer,
    model_name: str,
    messages: Sequence[dict[str, Any]] | None = None,
) -> Generator[Span, None, None]:
    """Span context manager for an individual LLM turn."""
    attributes: dict[str, Any] = {
        ATTR_LLM_MODEL: model_name,
    }
    if messages is not None:
        attributes[ATTR_LLM_PROMPT] = _safe_json_dumps(messages)

    with tracer.start_as_current_span(
        SPAN_LLM_CALL,
        attributes=attributes,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


@contextmanager
def trace_tool_call(
    tracer: Tracer,
    tool_name: str,
    arguments: Any = None,
) -> Generator[Span, None, None]:
    """Span context manager for a single tool execution."""
    attributes: dict[str, Any] = {
        ATTR_TOOL_NAME: tool_name,
    }
    if arguments is not None:
        attributes[ATTR_TOOL_INPUT] = _safe_json_dumps(arguments)

    with tracer.start_as_current_span(
        SPAN_TOOL_CALL,
        attributes=attributes,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute(ATTR_TOOL_ERROR, str(exc))
            span.set_status(StatusCode.ERROR, str(exc))
            raise
