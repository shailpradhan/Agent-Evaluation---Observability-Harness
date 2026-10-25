"""Run one task through agent execution, trace replay, judging, and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Protocol
from uuid import uuid4

from agent_eval.agent.agent import Agent
from agent_eval.agent.models import AgentResult
from agent_eval.judge.judge import BinaryRubricJudge
from agent_eval.models.judge import JudgeResult
from agent_eval.models.metrics import EvalResult, EvalRun, RegressionReport
from agent_eval.models.trajectory import Trajectory
from agent_eval.replay.engine import ReplayEngine, ReplayOutcome
from agent_eval.schema.task import Task
from agent_eval.tracing import TraceRunner


@dataclass
class EvaluationOutcome:
    """Artifacts and final score from one complete task evaluation."""

    task: Task
    agent_result: AgentResult
    trace: Trajectory
    replay: ReplayOutcome
    judge_result: JudgeResult
    score: EvalResult


@dataclass
class DatasetEvaluationOutcome:
    """Results and aggregate metrics from evaluating a complete dataset."""

    run: EvalRun
    results: list[EvalResult]
    outcomes: list[EvaluationOutcome]


class EvaluationStore(Protocol):
    """Persistence interface used by the dataset evaluation pipeline."""

    def save_dataset_evaluation(
        self, outcome: DatasetEvaluationOutcome
    ) -> None: ...


class EvaluationPipeline:
    """Evaluate one task or a full dataset with the configured agent."""

    def __init__(
        self,
        agent: Agent,
        replay_engine: ReplayEngine | None = None,
        judge: BinaryRubricJudge | None = None,
        trace_runner: TraceRunner | None = None,
        store: EvaluationStore | None = None,
    ) -> None:
        self.agent = agent
        self.replay_engine = replay_engine or ReplayEngine()
        self.judge = judge or BinaryRubricJudge()
        self.trace_runner = trace_runner or TraceRunner()
        self.store = store

    def evaluate(self, task: Task, *, run_id: str | None = None) -> EvaluationOutcome:
        """Execute, trace, replay, judge, and score one task."""
        evaluation_run_id = run_id or str(uuid4())
        traced_run = self.trace_runner.run(
            self.agent,
            task,
            run_id=evaluation_run_id,
        )
        agent_result = traced_run.agent_result
        trace = traced_run.trajectory

        replay = self.replay_engine.replay(self.agent, task, trace)
        replay_trace = replay.trajectory

        judge_result = self.judge.evaluate(
            task,
            replay_trace,
            run_id=evaluation_run_id,
        )
        resolved = bool(replay.agent_result.final_response.strip())
        score = EvalResult(
            result_id=str(uuid4()),
            run_id=evaluation_run_id,
            task_id=task.id,
            score=judge_result.overall_score,
            passed=judge_result.passed,
            latency_ms=trace.total_latency_ms,
            tokens=trace.total_tokens,
            resolved=resolved,
            tool_calls=len(trace.tool_calls),
            agent_turns=len(trace.llm_calls),
        )
        return EvaluationOutcome(
            task=task,
            agent_result=agent_result,
            trace=trace,
            replay=replay,
            judge_result=judge_result,
            score=score,
        )

    def evaluate_dataset(
        self,
        dataset: "EvaluationDataset",
        *,
        run_id: str | None = None,
        prompt_version: str = "unknown",
        metadata: dict[str, object] | None = None,
    ) -> DatasetEvaluationOutcome:
        """Evaluate every task, retaining task failures in the aggregate run."""
        from agent_eval.schema.dataset import EvaluationDataset

        if not isinstance(dataset, EvaluationDataset):
            raise TypeError("dataset must be an EvaluationDataset")
        if not dataset.tasks:
            raise ValueError("Cannot evaluate an empty dataset.")

        evaluation_run_id = run_id or str(uuid4())
        results: list[EvalResult] = []
        outcomes: list[EvaluationOutcome] = []
        for task in dataset.tasks:
            try:
                outcome = self.evaluate(task, run_id=evaluation_run_id)
            except Exception as exc:
                results.append(EvalResult(
                    result_id=str(uuid4()), run_id=evaluation_run_id,
                    task_id=task.id, score=0.0, passed=False, resolved=False,
                    error=f"{type(exc).__name__}: {exc}",
                ))
            else:
                outcomes.append(outcome)
                results.append(outcome.score)

        scores = [result.score for result in results]
        latencies = sorted(result.latency_ms for result in results)
        count = len(results)
        resolved = sum(result.resolved is True for result in results)
        model_version = getattr(self.agent.llm, "model_name", type(self.agent.llm).__name__)
        run = EvalRun(
            run_id=evaluation_run_id,
            model_version=str(model_version),
            prompt_version=prompt_version,
            dataset_version=dataset.version,
            mean_score=sum(scores) / count,
            pass_rate=sum(result.passed for result in results) / count,
            total_tasks=count,
            passed_tasks=sum(result.passed for result in results),
            failed_tasks=sum(not result.passed for result in results),
            resolved_tasks=resolved,
            total_tokens=sum(result.tokens for result in results),
            tokens_per_task=sum(result.tokens for result in results) / count,
            tool_calls_per_task=sum(result.tool_calls for result in results) / count,
            agent_turns_per_task=sum(result.agent_turns for result in results) / count,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            metadata={"dataset_name": dataset.name, **(metadata or {})},
        )
        outcome = DatasetEvaluationOutcome(run=run, results=results, outcomes=outcomes)
        if self.store is not None:
            self.store.save_dataset_evaluation(outcome)
        return outcome


def compare_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    baseline_results: list[EvalResult],
    candidate_results: list[EvalResult],
    *,
    score_drop_threshold: float = 0.05,
    pass_rate_drop_threshold: float = 0.05,
    task_score_drop_threshold: float = 0.05,
) -> RegressionReport:
    """Compare two same-dataset runs and report meaningful quality regressions."""
    if baseline.dataset_version != candidate.dataset_version:
        raise ValueError("Runs must use the same dataset version to be compared.")
    for name, threshold in (
        ("score_drop_threshold", score_drop_threshold),
        ("pass_rate_drop_threshold", pass_rate_drop_threshold),
        ("task_score_drop_threshold", task_score_drop_threshold),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    baseline_by_task = _unique_results(baseline_results, baseline.run_id)
    candidate_by_task = _unique_results(candidate_results, candidate.run_id)
    if (
        len(baseline_by_task) != baseline.total_tasks
        or len(candidate_by_task) != candidate.total_tasks
    ):
        raise ValueError("Each run must include a result for every evaluated task.")
    if baseline_by_task.keys() != candidate_by_task.keys():
        raise ValueError("Runs must contain results for the same task IDs.")
    regressed = [
        task_id for task_id in baseline_by_task
        if baseline_by_task[task_id].score - candidate_by_task[task_id].score
        >= task_score_drop_threshold
        and baseline_by_task[task_id].score > candidate_by_task[task_id].score
    ]
    mean_diff = candidate.mean_score - baseline.mean_score
    pass_diff = candidate.pass_rate - baseline.pass_rate
    detected = (
        mean_diff <= -score_drop_threshold
        or pass_diff <= -pass_rate_drop_threshold
        or bool(regressed)
    )
    summary = (
        f"Mean score changed by {mean_diff:+.3f}; pass rate changed by {pass_diff:+.3f}. "
        f"{len(regressed)} task(s) regressed."
    )
    baseline_latency = baseline.p95_latency_ms or 0.0
    candidate_latency = candidate.p95_latency_ms or 0.0
    return RegressionReport(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        mean_score_diff=mean_diff,
        pass_rate_diff=pass_diff,
        latency_diff_ms=candidate_latency - baseline_latency,
        regression_detected=detected,
        regressed_tasks=regressed,
        summary=summary,
    )


def _unique_results(results: list[EvalResult], run_id: str) -> dict[str, EvalResult]:
    mapped: dict[str, EvalResult] = {}
    for result in results:
        if result.run_id != run_id:
            raise ValueError(f"Result for task '{result.task_id}' belongs to another run.")
        if result.task_id in mapped:
            raise ValueError(f"Duplicate result for task '{result.task_id}'.")
        mapped[result.task_id] = result
    return mapped


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile from an already sorted nonempty list."""
    return sorted_values[max(0, ceil(len(sorted_values) * percentile) - 1)]
