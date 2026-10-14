"""Tests for OpenTelemetry instrumentation and spans."""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent_eval.agent.agent import Agent
from agent_eval.agent.demo import create_default_tools
from agent_eval.agent.llm import FakeLLM
from agent_eval.schema.task import Task
from agent_eval.telemetry import (
    ATTR_AGENT_FINAL_RESPONSE,
    ATTR_AGENT_MODEL,
    ATTR_AGENT_RUN_ID,
    ATTR_AGENT_TASK_ID,
    ATTR_LLM_LATENCY_MS,
    ATTR_LLM_MODEL,
    ATTR_LLM_PROMPT,
    ATTR_LLM_RESPONSE,
    ATTR_TOOL_ERROR,
    ATTR_TOOL_INPUT,
    ATTR_TOOL_LATENCY_MS,
    ATTR_TOOL_NAME,
    ATTR_TOOL_OUTPUT,
    SPAN_AGENT_RUN,
    SPAN_LLM_CALL,
    SPAN_TOOL_CALL,
    get_tracer,
    reset_telemetry,
    setup_telemetry,
    trace_agent_run,
    trace_llm_call,
    trace_tool_call,
)
from agent_eval.tracing import TraceRunner


@pytest.fixture(autouse=True)
def clean_telemetry():
    """Ensure clean telemetry setup for each test."""
    reset_telemetry()
    yield
    reset_telemetry()


@pytest.fixture
def memory_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    setup_telemetry(service_name="test-agent-eval", exporter=exporter)
    return exporter


def test_setup_telemetry_registers_provider(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")
    assert tracer is not None

    with tracer.start_as_current_span("simple-span") as span:
        span.set_attribute("test.key", "test.val")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "simple-span"
    assert spans[0].attributes["test.key"] == "test.val"


def test_trace_agent_run_emits_expected_span(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")

    with trace_agent_run(tracer, run_id="run-123", task_id="task-456", model_name="gpt-4") as span:
        span.set_attribute(ATTR_AGENT_FINAL_RESPONSE, "Order refunded successfully")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span_data = spans[0]
    assert span_data.name == SPAN_AGENT_RUN
    assert span_data.attributes[ATTR_AGENT_RUN_ID] == "run-123"
    assert span_data.attributes[ATTR_AGENT_TASK_ID] == "task-456"
    assert span_data.attributes[ATTR_AGENT_MODEL] == "gpt-4"
    assert span_data.attributes[ATTR_AGENT_FINAL_RESPONSE] == "Order refunded successfully"


def test_trace_llm_call_emits_expected_span(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")
    messages = [{"role": "user", "content": "Refund order 1234"}]

    with trace_llm_call(tracer, model_name="fake-llm", messages=messages) as span:
        span.set_attribute(ATTR_LLM_RESPONSE, '{"tool_call": {"name": "lookup_order"}}')
        span.set_attribute(ATTR_LLM_LATENCY_MS, 45.2)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span_data = spans[0]
    assert span_data.name == SPAN_LLM_CALL
    assert span_data.attributes[ATTR_LLM_MODEL] == "fake-llm"
    assert json.loads(span_data.attributes[ATTR_LLM_PROMPT]) == messages
    assert "lookup_order" in span_data.attributes[ATTR_LLM_RESPONSE]
    assert span_data.attributes[ATTR_LLM_LATENCY_MS] == 45.2


def test_trace_tool_call_emits_expected_span(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")
    tool_args = {"order_id": "1234"}

    with trace_tool_call(tracer, tool_name="lookup_order", arguments=tool_args) as span:
        span.set_attribute(ATTR_TOOL_OUTPUT, '{"order_id": "1234", "status": "shipped"}')
        span.set_attribute(ATTR_TOOL_LATENCY_MS, 12.5)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span_data = spans[0]
    assert span_data.name == SPAN_TOOL_CALL
    assert span_data.attributes[ATTR_TOOL_NAME] == "lookup_order"
    assert json.loads(span_data.attributes[ATTR_TOOL_INPUT]) == tool_args
    assert "shipped" in span_data.attributes[ATTR_TOOL_OUTPUT]
    assert span_data.attributes[ATTR_TOOL_LATENCY_MS] == 12.5


def test_trace_tool_call_records_exception(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")

    with (
        pytest.raises(ValueError, match="Database error"),
        trace_tool_call(tracer, tool_name="failing_tool", arguments={"id": 1}),
    ):
        raise ValueError("Database error")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span_data = spans[0]
    assert span_data.name == SPAN_TOOL_CALL
    assert span_data.status.status_code == StatusCode.ERROR
    assert span_data.attributes[ATTR_TOOL_ERROR] == "Database error"


def test_trace_runner_emits_full_span_tree(memory_exporter: InMemorySpanExporter):
    tracer = get_tracer("test_tracer")
    agent = Agent(
        llm=FakeLLM(),
        tools=create_default_tools(),
        approval_handler=lambda name, args: True,
    )
    task = Task(id="test_refund", prompt="Refund order 1234")

    runner = TraceRunner(tracer=tracer)
    traced_run = runner.run(agent, task, run_id="run-full-otel")

    assert traced_run.agent_result is not None
    assert "refund for order 1234 has been processed" in traced_run.agent_result.final_response

    spans = memory_exporter.get_finished_spans()
    assert len(spans) > 0

    span_names = [span.name for span in spans]
    # Root agent run span + LLM spans + Tool spans
    assert SPAN_AGENT_RUN in span_names
    assert SPAN_LLM_CALL in span_names
    assert SPAN_TOOL_CALL in span_names

    agent_run_span = next(s for s in spans if s.name == SPAN_AGENT_RUN)
    assert agent_run_span.attributes[ATTR_AGENT_RUN_ID] == "run-full-otel"
    assert agent_run_span.attributes[ATTR_AGENT_TASK_ID] == "test_refund"
    assert ATTR_AGENT_FINAL_RESPONSE in agent_run_span.attributes

    # Verify tool spans recorded tool names and inputs
    tool_spans = [s for s in spans if s.name == SPAN_TOOL_CALL]
    assert len(tool_spans) == 3
    tool_names = [s.attributes[ATTR_TOOL_NAME] for s in tool_spans]
    assert tool_names == ["lookup_order", "check_refund_eligibility", "refund_order"]

    # Verify child spans share the parent trace_id
    trace_id = agent_run_span.context.trace_id
    for span in spans:
        assert span.context.trace_id == trace_id
