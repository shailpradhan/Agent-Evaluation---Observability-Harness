"""Execution tracing wrappers kept outside the core agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from opentelemetry.trace import Tracer

from agent_eval.agent.agent import Agent
from agent_eval.agent.llm import LLM
from agent_eval.agent.models import AgentResult, LLMDecision
from agent_eval.agent.tools import Tool, ToolRegistry
from agent_eval.models.trajectory import LLMCall, ToolCall, ToolResponse, Trajectory
from agent_eval.schema.task import Task
from agent_eval.telemetry.spans import (
    ATTR_AGENT_FINAL_RESPONSE,
    ATTR_LLM_LATENCY_MS,
    ATTR_LLM_RESPONSE,
    ATTR_TOOL_LATENCY_MS,
    ATTR_TOOL_OUTPUT,
    _safe_json_dumps,
    trace_agent_run,
    trace_llm_call,
    trace_tool_call,
)
from agent_eval.telemetry.tracer import get_tracer


@dataclass
class TracedRun:
    """Agent result and trajectory collected by the evaluation layer."""

    agent_result: AgentResult
    trajectory: Trajectory


class _TracingLLM(LLM):
    def __init__(
        self,
        wrapped: LLM,
        trajectory: Trajectory,
        tracer: Tracer | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._trajectory = trajectory
        self._tracer = tracer
        self.model_name = getattr(wrapped, "model_name", type(wrapped).__name__)

    def generate(self, messages: list[dict[str, str]]) -> LLMDecision:
        tracer = self._tracer or get_tracer()
        with trace_llm_call(tracer, self.model_name, messages) as span:
            started = perf_counter()
            decision = self._wrapped.generate(messages)
            latency_ms = (perf_counter() - started) * 1000

            response_json = decision.model_dump_json()
            if span.is_recording():
                span.set_attribute(ATTR_LLM_RESPONSE, response_json)
                span.set_attribute(ATTR_LLM_LATENCY_MS, latency_ms)

            self._trajectory.record_llm_call(
                LLMCall(
                    model=self.model_name,
                    prompt=[dict(message) for message in messages],
                    response=response_json,
                    latency_ms=latency_ms,
                )
            )
            return decision


class TraceRunner:
    """Wrap an agent's LLM and tools to capture a trajectory and OpenTelemetry spans."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or get_tracer()

    def run(
        self,
        agent: Agent,
        task: Task,
        *,
        run_id: str | None = None,
    ) -> TracedRun:
        execution_run_id = run_id or str(uuid4())
        model_name = getattr(agent.llm, "model_name", type(agent.llm).__name__)

        trajectory = Trajectory(
            run_id=execution_run_id,
            task_id=task.id,
            model_name=model_name,
        )
        tools = (
            agent.tools.tools
            if isinstance(agent.tools, ToolRegistry)
            else tuple(agent.tools)
        )
        traced_tools = [
            self._trace_tool(tool, trajectory, self.tracer) for tool in tools
        ]
        traced_agent = Agent(
            llm=_TracingLLM(agent.llm, trajectory, self.tracer),
            tools=traced_tools,
            max_iterations=agent.max_iterations,
            approval_handler=agent.approval_handler,
        )

        with trace_agent_run(
            self.tracer, execution_run_id, task.id, model_name
        ) as run_span:
            result = traced_agent.run(task.prompt)
            if run_span.is_recording():
                run_span.set_attribute(
                    ATTR_AGENT_FINAL_RESPONSE, result.final_response
                )

        trajectory.set_final_response(result.final_response)
        return TracedRun(agent_result=result, trajectory=trajectory)

    @staticmethod
    def _trace_tool(
        tool: Tool[Any, Any],
        trajectory: Trajectory,
        tracer: Tracer | None = None,
    ) -> Tool[Any, Any]:
        ot_tracer = tracer or get_tracer()

        def traced_function(arguments: Any) -> Any:
            input_data = arguments.model_dump(mode="json")
            trajectory.record_tool_call(
                ToolCall(tool=tool.name, arguments=input_data)
            )

            with trace_tool_call(ot_tracer, tool.name, input_data) as span:
                started = perf_counter()
                try:
                    output = tool.output_model.model_validate(
                        tool.function(arguments)
                    )
                except Exception as exc:
                    latency_ms = (perf_counter() - started) * 1000
                    if span.is_recording():
                        span.set_attribute(ATTR_TOOL_LATENCY_MS, latency_ms)
                    trajectory.record_tool_response(
                        ToolResponse(
                            tool=tool.name,
                            input=input_data,
                            error=str(exc),
                            latency_ms=latency_ms,
                        )
                    )
                    raise

                latency_ms = (perf_counter() - started) * 1000
                output_data = output.model_dump(mode="json")
                if span.is_recording():
                    span.set_attribute(
                        ATTR_TOOL_OUTPUT, _safe_json_dumps(output_data)
                    )
                    span.set_attribute(ATTR_TOOL_LATENCY_MS, latency_ms)

                trajectory.record_tool_response(
                    ToolResponse(
                        tool=tool.name,
                        input=input_data,
                        output=output_data,
                        latency_ms=latency_ms,
                    )
                )
                return output

        return Tool(
            name=tool.name,
            description=tool.description,
            function=traced_function,
            input_model=tool.input_model,
            output_model=tool.output_model,
            requires_approval=tool.requires_approval,
            final_response_from_result=tool.final_response_from_result,
        )
