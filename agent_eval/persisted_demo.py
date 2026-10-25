"""Run the bundled refund dataset and persist results to PostgreSQL."""

from __future__ import annotations

import os
from pathlib import Path

from agent_eval.agent import Agent, FakeLLM, create_default_tools
from agent_eval.evaluation import EvaluationPipeline
from agent_eval.schema.dataset import EvaluationDataset
from agent_eval.storage import PostgresStore


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Set DATABASE_URL to connect to PostgreSQL.")

    store = PostgresStore(dsn)
    store.initialize()
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "refund.yaml"
    dataset = EvaluationDataset.load_from_yaml(dataset_path)
    agent = Agent(
        llm=FakeLLM(),
        tools=create_default_tools(),
        approval_handler=lambda _tool_name, _arguments: True,
    )
    outcome = EvaluationPipeline(agent, store=store).evaluate_dataset(
        dataset,
        prompt_version="demo-v1",
        metadata={"execution": "docker-compose"},
    )

    print(f"Persisted evaluation run: {outcome.run.run_id}")
    print(
        f"Tasks: {outcome.run.total_tasks}; passed: {outcome.run.passed_tasks}; "
        f"mean score: {outcome.run.mean_score:.3f}; "
        f"pass rate: {outcome.run.pass_rate:.1%}"
    )
    print("Task results and successful execution artifacts are stored in PostgreSQL.")


if __name__ == "__main__":
    main()
