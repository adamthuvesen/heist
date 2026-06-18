from __future__ import annotations

from pathlib import Path

import polars as pl

from heist.models import TaskRunResult
from heist.usage import cost_source_label as _cost_source
from heist.usage import primary_cost as _primary_cost


def export_eval_audit(run_dir: Path, results: list[TaskRunResult]) -> Path:
    if not results:
        # pl.DataFrame([]) writes a schema-less, zero-column parquet that the CLI
        # would report as a success — fail loudly instead.
        raise ValueError(f"run {run_dir} has no results to export")
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
    pl.DataFrame(rows).write_parquet(path)
    return path
