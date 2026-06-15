"""Helpers for building synthetic run dirs in tests.

Light enough to use in cross-run, baseline, and reporting tests without
invoking the full runner. The shapes match what `run_benchmark` would
write, so consumers see the same data as a real run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from heist.models import RUN_MANIFEST_SCHEMA_VERSION, RunManifest, TaskRunResult


def make_result(
    *,
    run_id: str,
    agent_id: str = "fake-pass",
    task_id: str = "marker",
    score: float = 1.0,
    success: bool | None = True,
    latency_s: float | None = 1.0,
    cost_usd: float | None = 0.01,
    outcome_status: str = "graded",
    agent_label: str | None = None,
    model_id: str = "fake-model",
) -> TaskRunResult:
    return TaskRunResult(
        run_id=run_id,
        agent_id=agent_id,
        agent_label=agent_label or agent_id,
        model_id=model_id,
        suite="smoke",
        task_id=task_id,
        task_title=task_id,
        task_category="fake",
        success=success,
        partial_credit=score if outcome_status == "graded" else None,
        outcome_status=outcome_status,  # type: ignore[arg-type]
        score=score,
        checks=[],
        latency_s=latency_s,
        cost_usd=cost_usd,
        workspace_path=f"/tmp/{run_id}/{agent_id}/{task_id}",
        diff_path=f"/tmp/{run_id}/{agent_id}/{task_id}/diff.patch",
        grader_path=f"/tmp/{run_id}/{agent_id}/{task_id}/grader.json",
        stdout_path=f"/tmp/{run_id}/{agent_id}/{task_id}/stdout.txt",
        stderr_path=f"/tmp/{run_id}/{agent_id}/{task_id}/stderr.txt",
    )


def write_synthetic_run(
    runs_dir: Path,
    run_id: str,
    *,
    suite: str = "smoke",
    agent_ids: list[str] | None = None,
    task_ids: list[str] | None = None,
    results: list[TaskRunResult] | None = None,
    created_at: datetime | None = None,
    harness_git_sha: str | None = "0" * 40,
    tags: list[str] | None = None,
    kind: str = "live",
    source_run_id: str | None = None,
    status: str = "completed",
) -> Path:
    """Write a `runs/<run_id>/{manifest.json,results.jsonl}` pair.

    Defaults: one pass row for (fake-pass, marker). Override `results` for
    bespoke data; agent_ids / task_ids are derived from results when both are
    omitted.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if results is None:
        results = [make_result(run_id=run_id)]
    if agent_ids is None:
        agent_ids = sorted({r.agent_id for r in results})
    if task_ids is None:
        task_ids = sorted({r.task_id for r in results})

    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        suite=suite,
        agent_ids=list(agent_ids),
        task_ids=list(task_ids),
        created_at=created_at or datetime.now(UTC),
        completed_at=created_at or datetime.now(UTC),
        duration_s=1.0,
        repo_root=str(runs_dir.parent),
        run_dir=str(run_dir),
        default_agents=[],
        status=status,  # type: ignore[arg-type]
        harness_git_sha=harness_git_sha,
        tags=list(tags or []),
        kind=kind,  # type: ignore[arg-type]
        source_run_id=source_run_id,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    with (run_dir / "results.jsonl").open("w") as handle:
        for result in results:
            handle.write(result.model_dump_json())
            handle.write("\n")
    return run_dir


def write_two_runs(
    runs_dir: Path,
    *,
    agent_id: str = "fake-pass",
    task_id: str = "marker",
) -> tuple[str, str]:
    """Convenience for tests that need a baseline + follow-up pair, with
    `run-b` chronologically after `run-a`. Returns (older, newer)."""
    older_at = datetime.now(UTC) - timedelta(hours=1)
    newer_at = datetime.now(UTC)
    write_synthetic_run(
        runs_dir,
        "run-a",
        results=[
            make_result(
                run_id="run-a",
                agent_id=agent_id,
                task_id=task_id,
                score=0.8,
                cost_usd=0.10,
                latency_s=2.0,
            )
        ],
        created_at=older_at,
        harness_git_sha="a" * 40,
    )
    write_synthetic_run(
        runs_dir,
        "run-b",
        results=[
            make_result(
                run_id="run-b",
                agent_id=agent_id,
                task_id=task_id,
                score=0.5,
                cost_usd=0.20,
                latency_s=3.0,
            )
        ],
        created_at=newer_at,
        harness_git_sha="b" * 40,
    )
    return "run-a", "run-b"


def write_corrupt_run(runs_dir: Path, run_id: str = "broken") -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("not json")
    return run_dir


__all__ = [
    "make_result",
    "write_synthetic_run",
    "write_two_runs",
    "write_corrupt_run",
]


# Importing json here keeps the module imports tidy when tests construct
# bespoke manifest payloads (e.g., to write a v1 manifest manually).
_ = json
