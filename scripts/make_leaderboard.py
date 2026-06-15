"""Reduce a HEIST run's results.jsonl to a sanitized, leak-free leaderboard.jsonl.

The published leaderboard keeps only an allowlist of non-sensitive columns. It
deliberately drops the hidden grader's per-check breakdown (``checks``), the
error detail (``error``), and every absolute filesystem path (the ``*_path``
columns) — none of which can appear in a public artifact without leaking grader
expectations or a local home directory.

    source: runs/<run-id>/results.jsonl        (git-ignored, never published)
    output: results/<run-id>/leaderboard.jsonl (local only — not committed here)

This produces a local leak-free leaderboard for your own analysis. The public
repo no longer commits it; the published artifact is the aggregate report
(results/<run>/report.html), guarded by tests/test_report_leakfree.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/make_leaderboard.py` to import the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heist.tasks import load_tasks  # noqa: E402

# The exact columns that survive into the published leaderboard. Anything not
# named here — including `checks`, `error`, and every `*_path` field — is
# dropped. The leak-guard test pins this same set independently.
LEADERBOARD_KEYS = (
    "agent",
    "model",
    "task",
    "score",
    "success",
    "latency_s",
    "cost_usd",
    "task_category",
)

# Published column -> source column it is read from. Renames the verbose run
# columns (agent_label, model_id, task_id) to the public leaderboard names.
SOURCE_KEYS = {
    "agent": "agent_label",
    "model": "model_id",
    "task": "task_id",
    "score": "score",
    "success": "success",
    "latency_s": "latency_s",
    "cost_usd": "cost_usd",
    "task_category": "task_category",
}


def reduce_row(row: dict) -> dict:
    """Project one results.jsonl row onto the allowlist, failing loudly on gaps."""
    missing = [src for src in SOURCE_KEYS.values() if src not in row]
    if missing:
        raise ValueError(f"results row missing required columns: {missing}")
    reduced = {key: row[src] for key, src in SOURCE_KEYS.items()}
    reduced["success"] = bool(reduced["success"])
    return reduced


def build_leaderboard(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    with results_path.open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{results_path}:{lineno}: invalid JSON: {exc}") from exc
            rows.append(reduce_row(row))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize a run into a published leaderboard.")
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/frontier-2026-05-17"),
        help="run directory containing results.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: results/<run-name>/leaderboard.jsonl)",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help=(
            "restrict the published board to the tasks currently in this suite "
            "(e.g. 'frontier' or 'calibration'). Rows for tasks outside the suite "
            "are dropped, so retiring a task to another tier drops it from the board."
        ),
    )
    args = parser.parse_args()

    results_path = args.run / "results.jsonl"
    if not results_path.is_file():
        parser.error(f"no results.jsonl under {args.run}")

    out_path = args.out or Path("results") / args.run.name / "leaderboard.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_leaderboard(results_path)

    if args.suite:
        suite_ids = {task.id for task in load_tasks(args.suite)}
        kept = [row for row in rows if row["task"] in suite_ids]
        dropped = sorted({row["task"] for row in rows} - suite_ids)
        if dropped:
            print(
                f"dropped {len(dropped)} task(s) not in suite {args.suite!r}: {', '.join(dropped)}"
            )
        rows = kept
    with out_path.open("w") as handle:
        for row in rows:
            if set(row) != set(LEADERBOARD_KEYS):
                raise ValueError(f"row escaped the allowlist: {sorted(row)}")
            handle.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
