"""PostgreSQL persistence for evaluation runs, results, and task artifacts."""

from __future__ import annotations

from typing import Any

from agent_eval.evaluation.pipeline import DatasetEvaluationOutcome
from agent_eval.models.metrics import EvalResult, EvalRun, RegressionReport


class PostgresStore:
    """Persist complete dataset evaluations using PostgreSQL JSONB payloads.

    Relational columns support common run/result lookups. JSONB payload columns
    retain the complete Pydantic models and execution artifacts for later use.
    Each save is one transaction, so a run is never stored partially.
    """

    def __init__(self, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("A PostgreSQL DSN is required.")
        self.dsn = dsn

    def initialize(self) -> None:
        """Create the storage tables if they do not already exist."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    mean_score DOUBLE PRECISION NOT NULL,
                    pass_rate DOUBLE PRECISION NOT NULL,
                    total_tasks INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                    task_id TEXT NOT NULL,
                    score DOUBLE PRECISION NOT NULL,
                    passed BOOLEAN NOT NULL,
                    payload JSONB NOT NULL,
                    UNIQUE (run_id, task_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_artifacts (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id, task_id)
                        REFERENCES eval_results(run_id, task_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS regression_reports (
                    baseline_run_id TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                    candidate_run_id TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                    regression_detected BOOLEAN NOT NULL,
                    payload JSONB NOT NULL,
                    PRIMARY KEY (baseline_run_id, candidate_run_id)
                )
                """
            )

    def save_dataset_evaluation(
        self, outcome: DatasetEvaluationOutcome
    ) -> None:
        """Save run metrics, every task result, and all successful task artifacts."""
        from psycopg.types.json import Jsonb

        run = outcome.run
        results_by_task = {result.task_id: result for result in outcome.results}
        if len(results_by_task) != len(outcome.results):
            raise ValueError("A dataset evaluation cannot contain duplicate task results.")
        if len(outcome.results) != run.total_tasks:
            raise ValueError("The run must have one result for every evaluated task.")
        if any(result.run_id != run.run_id for result in outcome.results):
            raise ValueError("Every task result must belong to the evaluation run.")

        artifact_payloads: dict[str, dict[str, Any]] = {}
        for task_outcome in outcome.outcomes:
            task_id = task_outcome.task.id
            if task_id not in results_by_task:
                raise ValueError(f"Task artifact '{task_id}' has no evaluation result.")
            artifact_payloads[task_id] = {
                "task": task_outcome.task.model_dump(mode="json"),
                "agent_result": task_outcome.agent_result.model_dump(mode="json"),
                "trajectory": task_outcome.trace.model_dump(mode="json"),
                "replay": {
                    "session": task_outcome.replay.session.model_dump(mode="json"),
                    "agent_result": task_outcome.replay.agent_result.model_dump(mode="json"),
                    "trajectory": task_outcome.replay.trajectory.model_dump(mode="json"),
                },
                "judge_result": task_outcome.judge_result.model_dump(mode="json"),
            }

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO eval_runs (
                    run_id, dataset_version, model_version, prompt_version,
                    mean_score, pass_rate, total_tasks, created_at, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run.run_id, run.dataset_version, run.model_version,
                    run.prompt_version, run.mean_score, run.pass_rate,
                    run.total_tasks, run.timestamp, Jsonb(run.model_dump(mode="json")),
                ),
            )
            for result in outcome.results:
                connection.execute(
                    """
                    INSERT INTO eval_results (
                        result_id, run_id, task_id, score, passed, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.result_id, result.run_id, result.task_id,
                        result.score, result.passed,
                        Jsonb(result.model_dump(mode="json")),
                    ),
                )
            for task_id, payload in artifact_payloads.items():
                connection.execute(
                    "INSERT INTO eval_artifacts (run_id, task_id, payload) "
                    "VALUES (%s, %s, %s)",
                    (run.run_id, task_id, Jsonb(payload)),
                )

    def get_run(self, run_id: str) -> EvalRun | None:
        """Load one aggregate run by its identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM eval_runs WHERE run_id = %s", (run_id,)
            ).fetchone()
        return EvalRun.model_validate(row[0]) if row else None

    def list_runs(self, limit: int = 50) -> list[EvalRun]:
        """Return the newest run summaries first."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM eval_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [EvalRun.model_validate(row[0]) for row in rows]

    def get_results(self, run_id: str) -> list[EvalResult]:
        """Load all per-task results for a run in insertion order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM eval_results WHERE run_id = %s ORDER BY task_id",
                (run_id,),
            ).fetchall()
        return [EvalResult.model_validate(row[0]) for row in rows]

    def get_artifacts(self, run_id: str, task_id: str) -> dict[str, Any] | None:
        """Load full task, trace, replay, and judge artifacts when available."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT payload FROM eval_artifacts
                   WHERE run_id = %s AND task_id = %s""",
                (run_id, task_id),
            ).fetchone()
        return row[0] if row else None

    def save_regression_report(self, report: RegressionReport) -> None:
        """Persist a comparison report for two runs already stored in this database."""
        from psycopg.types.json import Jsonb

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO regression_reports (
                       baseline_run_id, candidate_run_id, regression_detected, payload
                   ) VALUES (%s, %s, %s, %s)""",
                (
                    report.baseline_run_id,
                    report.candidate_run_id,
                    report.regression_detected,
                    Jsonb(report.model_dump(mode="json")),
                ),
            )

    def get_regression_report(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> RegressionReport | None:
        """Load a stored comparison report."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT payload FROM regression_reports
                   WHERE baseline_run_id = %s AND candidate_run_id = %s""",
                (baseline_run_id, candidate_run_id),
            ).fetchone()
        return RegressionReport.model_validate(row[0]) if row else None

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)


__all__ = ["PostgresStore"]
