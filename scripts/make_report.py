"""Re-render a run's committed summary.md + report.html from its results.jsonl.

This is the report counterpart to scripts/make_leaderboard.py: it reads a run's
``results.jsonl`` (git-ignored, local-only) and writes the published, committed
report artifacts under ``results/<run-name>/``.

    source: runs/<run-id>/results.jsonl        (git-ignored, never published)
    output: results/<run-name>/summary.md      (committed snapshot)
            results/<run-name>/report.html

Pass ``--suite`` to restrict the report to the tasks currently in that suite, so
retiring a task to another tier (e.g. moving a saturated canary from
``tasks/frontier`` to ``tasks/calibration``) drops it from the published board on
the next regeneration. The headline metric is alpha (α); see heist.reporting.

A small compatibility shim renames the one legacy results column
(``reported_run_total_cost_usd`` -> ``reported_session_cost_usd``) and drops keys
the current model no longer carries, so an older snapshot still re-renders.

    uv run python scripts/make_report.py --run runs/frontier-2026-06-15 --suite frontier
    make report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/make_report.py` to import the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heist.models import TaskRunResult  # noqa: E402
from heist.reporting import write_report  # noqa: E402
from heist.tasks import load_tasks  # noqa: E402

# Legacy results columns -> current model field (or dropped when unmapped). Keeps
# a pre-rename snapshot loadable without a one-off migration of the raw run.
_LEGACY_RENAMES = {"reported_run_total_cost_usd": "reported_session_cost_usd"}


def _coerce_row(row: dict) -> TaskRunResult:
    migrated = {_LEGACY_RENAMES.get(key, key): value for key, value in row.items()}
    known = {key: value for key, value in migrated.items() if key in TaskRunResult.model_fields}
    return TaskRunResult.model_validate(known)


def load_results_tolerant(results_path: Path) -> list[TaskRunResult]:
    results: list[TaskRunResult] = []
    with results_path.open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{results_path}:{lineno}: invalid JSON: {exc}") from exc
            results.append(_coerce_row(row))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-render a run's published report artifacts.")
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/frontier-2026-06-15"),
        help="run directory containing results.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: results/<run-name>/)",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="restrict the report to tasks currently in this suite (e.g. 'frontier').",
    )
    args = parser.parse_args()

    results_path = args.run / "results.jsonl"
    if not results_path.is_file():
        parser.error(f"no results.jsonl under {args.run}")

    out_dir = args.out_dir or Path("results") / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_tolerant(results_path)

    if args.suite:
        suite_ids = {task.id for task in load_tasks(args.suite)}
        dropped = sorted({r.task_id for r in results} - suite_ids)
        if dropped:
            print(
                f"dropped {len(dropped)} task(s) not in suite {args.suite!r}: {', '.join(dropped)}"
            )
        results = [r for r in results if r.task_id in suite_ids]

    if not results:
        parser.error("no results left to render after filtering")

    n_tasks = len({r.task_id for r in results})
    write_report(out_dir, results)
    print(f"wrote {out_dir}/summary.md + report.html ({len(results)} rows, {n_tasks} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
