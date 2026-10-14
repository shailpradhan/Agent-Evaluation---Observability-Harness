# Agent Evaluation & Observability Harness

An end-to-end testing, observability, and evaluation platform for AI agents. It instruments multi-step agent workflows via OpenTelemetry, enables deterministic offline replay using recorded mock tools, and evaluates trajectory correctness with structured LLM-as-a-judge rubrics to prevent regressions across model, prompt, and tool iterations.

## Core Lifecycle

```
Task ➔ AI Agent ➔ Trajectory Capture with Open Telemetry ➔ Record ➔ Deterministic Replay Engine (MOCK TOOLS) ➔ Evaluate (Binary Rubric Judge) ➔ Regression Testing ➔ DB Persistence ➔ Artifacts Versioning
```

---

## Architecture Overview

Core data contracts, schema definitions, and directory structure:

1. **Project Structure**: Modular package layout supporting agent definitions, OpenTelemetry instrumentation, task corpus management, trajectory replay, LLM judge, and metrics persistence.
2. **Data Models (`agent_eval.models`)**:
   - `Trajectory`, `TrajectoryStep`, `ToolCall`, `ToolResponse`, `LLMCall`: Full agent execution recording.
   - `JudgeResult`, `RubricEvaluation`, `DimensionScores`: Structured judge results with each rubric item evaluated as one atomic yes/no assertion; aggregate scores remain separate.
   - `EvalRun`, `EvalResult`, `RegressionReport`: Run metrics, pass rates, latency/token metrics, and regression detection. Results include optional safety/compliance, resolution, and side-effect outcomes plus tool/turn counts; run aggregates include p50 latency and per-task efficiency/cost fields.
   - `DimensionScores`: Includes trajectory quality, safety/compliance, and side-effect scores. Tool-use scoring was already represented by `tool_usage`.
   - `MockToolEntry`, `ReplaySession`: Deterministic tool replay records.
3. **Task Schema (`agent_eval.schema`)**:
   - `Task`: Standard evaluation task specification (`id`, `prompt`, `expected_outcome`, `expected_tools`, `rubric`, `metadata`).
   - `EvaluationDataset`: Versioned task collection with YAML/JSON loader and serialization utilities.

---

## Directory Structure

```
Agent/
├── pyproject.toml              # Project dependencies and configuration
├── README.md                   # Harness overview and guide
├── datasets/                   # Versioned evaluation task corpus
│   ├── refund.yaml
│   └── cancellation.yaml
├── agent_eval/                 # Core harness package
│   ├── schema/                 # Task and dataset schemas
│   ├── models/                 # Trajectory, Judge, Metrics, and Replay models
│   ├── agent/                  # Agent implementation (Phase 1 next steps)
│   ├── telemetry/              # OpenTelemetry instrumentation (Phase 2)
│   ├── replay/                 # Trajectory replay engine (Phase 2)
│   ├── judge/                  # LLM judge engine (Phase 2)
│   ├── storage/                # SQLite/Postgres persistence (Phase 3)
│   └── evaluation/             # Test runner and regression detector (Phase 4)
└── tests/                      # Unit test suite
```

---

## Quickstart

### Running the Deterministic Agent Example

The first agent-layer implementation is provider-independent. `Agent` receives
an `LLM` implementation and a collection of tools; `FakeLLM` simulates only the
example refund flow and does not perform real reasoning or call an external API.

```powershell
python -m agent_eval.agent.demo
```

The example prints a final response and the ordered tool calls. The LLM and
tools are injected when constructing `Agent`, so a future provider adapter can
implement `LLM.generate(messages) -> LLMDecision` without changing the
orchestration code. The adapter translates its provider's native tool calls and
final text into this typed decision; the agent does not parse provider JSON.
Tool inputs and outputs are validated with Pydantic models.
Tool implementations receive their declared Pydantic input model and return
their declared Pydantic output model. The sample store includes delivered order
`1234` and cancelled order `5678`; unknown orders produce an explicit lookup
error, and refunds for an eligible order are idempotent within the process.
Refund processing requires an injected approval handler; the demo explicitly
auto-approves because its refund implementation only changes in-memory demo
state. Without an approval handler, the agent rejects protected calls. The
refund tool also formats its final status from the validated refund result, so
the LLM cannot override that outcome with an unsupported success claim.

The agent layer intentionally does not record telemetry or persist executions.
In the next phase, an outer adapter can observe LLM/tool boundaries and map the
returned `AgentResult` and call sequence into trajectory/trace data without
making the agent depend on OpenTelemetry, storage, or the replay engine.

### Running One Complete Evaluation

Run the refund example end to end:

```powershell
python -m agent_eval.demo
```

This executes one dataset task, captures its trajectory, replays its recorded
LLM decisions and tool results without calling the original tools, evaluates
the replay against the task's binary rubric, and returns an `EvalResult` score.
The demo uses the deterministic `FakeLLM`; it requires no API key or external
service.
