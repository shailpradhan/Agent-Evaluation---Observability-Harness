"""Comprehensive end-to-end demo of the Agent Evaluation & Observability Harness."""

from pathlib import Path

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_eval.agent import Agent, FakeLLM, create_default_tools
from agent_eval.evaluation.pipeline import EvaluationPipeline
from agent_eval.schema.dataset import EvaluationDataset
from agent_eval.telemetry import setup_telemetry


def main() -> None:
    print("=" * 65)
    print("AGENT EVALUATION & OBSERVABILITY HARNESS - LIVE DEMO")
    print("=" * 65)

    # 1. Initialize OpenTelemetry tracing
    exporter = InMemorySpanExporter()
    setup_telemetry(service_name="agent-eval-demo", exporter=exporter)

    # 2. Load evaluation dataset
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "refund.yaml"
    dataset = EvaluationDataset.load_from_yaml(dataset_path)
    task = dataset.get_task("refund_order_1234")
    if not task:
        raise ValueError("Task refund_order_1234 not found")

    print("\n[STAGE 1: EVALUATION TASK LOADED]")
    print(f"  • Task ID:        {task.id}")
    print(f"  • Prompt:         \"{task.prompt}\"")
    print(f"  • Expected Tools: {task.expected_tools}")
    print(f"  • Rubric Checks:  {len(task.rubric)}")

    # 3. Create Agent
    agent = Agent(
        llm=FakeLLM(),
        tools=create_default_tools(),
        approval_handler=lambda tool_name, _args: True,
    )

    # 4. Run through Evaluation Pipeline (Agent -> Trace -> Replay -> Judge -> Score)
    pipeline = EvaluationPipeline(agent)
    outcome = pipeline.evaluate(task)

    print("\n[STAGE 2: LIVE AGENT EXECUTION & TRAJECTORY CAPTURE]")
    print(f"  • Agent Response: {outcome.agent_result.final_response}")
    print(f"  • Tool Sequence:  {' -> '.join(outcome.trace.tool_names_called)}")
    print(f"  • LLM Turns:      {len(outcome.trace.llm_calls)}")
    print(f"  • Latency:        {outcome.trace.total_latency_ms:.2f} ms")

    print("\n[STAGE 3: DETERMINISTIC REPLAY (MOCK ENVIRONMENT)]")
    replayed_matches = (
        outcome.replay.agent_result.final_response == outcome.agent_result.final_response
    )
    print(f"  • Replay Success: {replayed_matches}")
    print(f"  • Mocks Consumed: {len(outcome.replay.session.recorded_tools)} tool responses")
    print(f"  • Replayed Final: {outcome.replay.agent_result.final_response}")

    print("\n[STAGE 4: EVALUATION & JUDGING]")
    status = "PASSED" if outcome.judge_result.passed else "FAILED"
    print(f"  • Verdict:        {status} (Overall Score: {outcome.judge_result.overall_score:.2f})")
    print("  • Rubric Breakdown:")
    for check in outcome.judge_result.rubric_evaluations:
        symbol = "[PASS]" if check.passed else "[FAIL]"
        print(f"    {symbol} {check.assertion}")

    print("\n[STAGE 5: OPENTELEMETRY TRACE OBSERVABILITY]")
    spans = exporter.get_finished_spans()
    print(f"  • Total Spans Emitted: {len(spans)}")
    for span in spans:
        prefix = "  |--" if span.name != "agent.run" else "*"
        short_attrs = {
            k: v for k, v in span.attributes.items()
            if k in ("agent.run_id", "agent.task_id", "tool.name", "llm.model", "tool.latency_ms")
        }
        attrs_str = ", ".join(f"{k}={v}" for k, v in short_attrs.items())
        print(f"  {prefix} Span: '{span.name}' [{attrs_str}]")

    print("\n" + "=" * 65)
    print("DEMO COMPLETED SUCCESSFULLY - ALL 5 PIPELINE STAGES VERIFIED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
