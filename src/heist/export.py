from __future__ import annotations

from pathlib import Path

import polars as pl

from heist.models import TaskRunResult
from heist.usage import cost_source_label as _cost_source
from heist.usage import primary_cost as _primary_cost


def export_eval_audit(run_dir: Path, results: list[TaskRunResult]) -> Path:
    out_dir = run_dir / "eval-audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "agent_id": result.agent_id,
                "model_id": result.model_id,
                "harness": "heist",
                "run_id": result.run_id,
                "task_id": result.task_id,
                "task_category": result.task_category,
                "seed": None,
                "success": result.success,
                "partial_credit": result.partial_credit
                if result.outcome_status == "graded"
                else None,
                "outcome_status": result.outcome_status,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "tokens_in_by_model": result.tokens_in_by_model,
                "tokens_out_by_model": result.tokens_out_by_model,
                "latency_s": result.latency_s,
                "timestamp": result.timestamp,
                "cost_usd": _primary_cost(result),
                "cost_source": _cost_source(result),
                "reconstructed_per_task_cost_usd": result.reconstructed_per_task_cost_usd,
                "reported_session_cost_usd": result.reported_session_cost_usd,
                "cost_provenance": result.cost_provenance,
                "rerun_metadata": {
                    "workspace_path": result.workspace_path,
                    "diff_path": result.diff_path,
                    "grader_path": result.grader_path,
                },
            }
        )
    path = out_dir / "runs.parquet"
    frame = pl.DataFrame(rows) if rows else _empty_audit_frame()
    frame.write_parquet(path)
    return path


def _empty_audit_frame() -> pl.DataFrame:
    """A zero-row frame carrying the full audit column set, so a run with no
    results still writes a typed, columned parquet instead of a schemaless 0-col
    one that breaks a downstream `select(["cost_usd", "success", ...])`. The
    template mirrors the row dict above; ``tests/test_export_and_report.py``
    asserts the empty and populated column sets match, catching any drift."""
    template = {
        "agent_id": "",
        "model_id": "",
        "harness": "heist",
        "run_id": "",
        "task_id": "",
        "task_category": "",
        "seed": None,
        "success": True,
        "partial_credit": 0.0,
        "outcome_status": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_in_by_model": {"": 0},
        "tokens_out_by_model": {"": 0},
        "latency_s": 0.0,
        "timestamp": "",
        "cost_usd": 0.0,
        "cost_source": "",
        "reconstructed_per_task_cost_usd": 0.0,
        "reported_session_cost_usd": 0.0,
        "cost_provenance": "",
        "rerun_metadata": {"workspace_path": "", "diff_path": "", "grader_path": ""},
    }
    return pl.DataFrame([template]).clear()
