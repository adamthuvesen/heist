"""Cross-run scanning, baseline tagging, comparison, and history.

The on-disk corpus under `runs/<run_id>/` is the source of truth. Every
function in this module reads from disk on demand — no index, no database.
Light caching keeps repeated reads inside a single CLI invocation cheap.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heist.models import RunKind, RunManifest, TaskRunResult
from heist.runner import load_manifest, load_results

logger = logging.getLogger("heist.history")

BASELINES_FILENAME = "baselines.json"
RESERVED_REFS: tuple[str, ...] = ("latest", "previous")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class HistoryError(RuntimeError):
    """Raised when a cross-run query cannot be satisfied."""


def _run_dir_for_id(runs_dir: Path, run_id: str) -> Path:
    """Return the on-disk run dir for a literal run id.

    Run references are ids, not paths. Rejecting separators keeps baseline
    tags and literal refs from escaping the configured runs directory.
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HistoryError(
            f"invalid run id {run_id!r}: use only letters, numbers, '.', '_', and '-'"
        )
    return runs_dir / run_id


def _run_exists(runs_dir: Path, run_id: str) -> bool:
    return (_run_dir_for_id(runs_dir, run_id) / "manifest.json").exists()


class CorruptRun(BaseModel):
    """A run dir whose manifest could not be parsed."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_dir: str
    reason: str


class RunSummary(BaseModel):
    """Per-run rollup used by `heist runs list` and comparison views."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    suite: str
    kind: RunKind
    source_run_id: str | None
    agent_ids: list[str]
    task_ids: list[str]
    created_at: datetime
    completed_at: datetime | None
    duration_s: float | None
    harness_git_sha: str | None
    tags: list[str]
    mean_score: float | None
    total_cost_usd: float | None
    row_count: int
    run_dir: str


# A run dir is identified by manifest.json. Anything else under runs/ is
# ignored (artifacts dirs, baselines.json, ad-hoc notes).
def _candidate_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    out: list[Path] = []
    for entry in sorted(runs_dir.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").exists():
            out.append(entry)
    return out


def _summary_from_manifest(manifest: RunManifest, results: list[TaskRunResult]) -> RunSummary:
    if results:
        scores = [row.score for row in results]
        costs = [row.cost_usd for row in results if row.cost_usd is not None]
        mean_score: float | None = sum(scores) / len(scores)
        total_cost: float | None = sum(costs) if costs else None
    else:
        mean_score = None
        total_cost = None
    return RunSummary(
        run_id=manifest.run_id,
        suite=manifest.suite,
        kind=manifest.kind,
        source_run_id=manifest.source_run_id,
        agent_ids=list(manifest.agent_ids),
        task_ids=list(manifest.task_ids),
        created_at=manifest.created_at,
        completed_at=manifest.completed_at,
        duration_s=manifest.duration_s,
        harness_git_sha=manifest.harness_git_sha,
        tags=list(manifest.tags),
        mean_score=mean_score,
        total_cost_usd=total_cost,
        row_count=len(results),
        run_dir=manifest.run_dir,
    )


@functools.lru_cache(maxsize=128)
def _cached_results(run_dir: str) -> tuple[TaskRunResult, ...]:
    return tuple(load_results(Path(run_dir)))


def load_run_results(runs_dir: Path, run_id: str) -> list[TaskRunResult]:
    """Read `results.jsonl` for a specific run, cached for the process."""
    run_dir = _run_dir_for_id(runs_dir, run_id)
    return list(_cached_results(str(run_dir.resolve())))


def load_all_runs(
    runs_dir: Path,
    *,
    include_corrupt: bool = False,
) -> tuple[list[RunSummary], list[CorruptRun]]:
    """Scan `runs_dir` and return per-run summaries (and any corrupt entries).

    Order: descending `created_at`. Falls back to the directory mtime when a
    manifest can't be loaded (those entries land in `corrupt` regardless).
    """
    summaries: list[RunSummary] = []
    corrupt: list[CorruptRun] = []
    for run_dir in _candidate_run_dirs(runs_dir):
        try:
            manifest = load_manifest(run_dir)
        except Exception as exc:
            corrupt.append(
                CorruptRun(
                    run_id=run_dir.name,
                    run_dir=str(run_dir),
                    reason=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
                )
            )
            logger.warning("could not load manifest at %s: %s", run_dir, exc)
            continue
        try:
            results = list(_cached_results(str(run_dir.resolve())))
        except FileNotFoundError:
            results = []
        except Exception as exc:
            logger.warning("could not load results.jsonl for %s: %s", manifest.run_id, exc)
            results = []
        summaries.append(_summary_from_manifest(manifest, results))
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    if include_corrupt:
        corrupt.sort(key=lambda c: c.run_id)
    else:
        corrupt = []
    return summaries, corrupt


def cross_run_table(
    runs_dir: Path,
    *,
    agent_id: str | None = None,
    task_id: str | None = None,
) -> list[dict[str, object]]:
    """Flat long-form view of every (run, agent, task) row.

    Sorted by created_at ascending so callers can render a timeline directly.
    Filters apply at row scan time, not after, so an unused filter doesn't pay
    for an O(N) scan it would discard.
    """
    summaries, _ = load_all_runs(runs_dir)
    summaries_by_id = {s.run_id: s for s in summaries}
    rows: list[dict[str, object]] = []
    for summary in summaries:
        if agent_id is not None and agent_id not in summary.agent_ids:
            continue
        if task_id is not None and task_id not in summary.task_ids:
            continue
        for result in load_run_results(runs_dir, summary.run_id):
            if agent_id is not None and result.agent_id != agent_id:
                continue
            if task_id is not None and result.task_id != task_id:
                continue
            rows.append(
                {
                    "run_id": summary.run_id,
                    "created_at": summaries_by_id[summary.run_id].created_at,
                    "agent_id": result.agent_id,
                    "agent_label": result.agent_label,
                    "task_id": result.task_id,
                    "score": result.score,
                    "success": result.success,
                    "outcome_status": result.outcome_status,
                    "latency_s": result.latency_s,
                    "cost_usd": result.cost_usd,
                    "harness_git_sha": summary.harness_git_sha,
                    "kind": summary.kind,
                }
            )
    rows.sort(key=lambda r: r["created_at"])  # type: ignore[arg-type, return-value]
    return rows


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


class BaselineRegistry(BaseModel):
    """Named pointers from tag → run_id, persisted to `runs/baselines.json`."""

    model_config = ConfigDict(extra="forbid")

    entries: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def load(cls, runs_dir: Path) -> BaselineRegistry:
        path = runs_dir / BASELINES_FILENAME
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise HistoryError(f"{path} is not a JSON object of tag → run_id")
        for key, value in raw.items():
            if not isinstance(value, str):
                raise HistoryError(
                    f"{path}: tag {key!r} maps to {type(value).__name__}, expected str"
                )
        return cls(entries=dict(raw))

    def list(self) -> dict[str, str]:
        return dict(self.entries)

    def get(self, tag: str) -> str | None:
        return self.entries.get(tag)

    def set(self, runs_dir: Path, run_id: str, tag: str) -> str | None:
        if tag in RESERVED_REFS:
            raise HistoryError(
                f"tag {tag!r} is reserved (resolves dynamically). Pick another name."
            )
        if not _run_exists(runs_dir, run_id):
            raise HistoryError(
                f"cannot set baseline {tag!r}: run {run_id!r} not found under {runs_dir}"
            )
        previous = self.entries.get(tag)
        self.entries[tag] = run_id
        self._save(runs_dir)
        return previous

    def unset(self, runs_dir: Path, tag: str) -> str:
        if tag not in self.entries:
            raise HistoryError(f"baseline tag {tag!r} is not defined")
        removed = self.entries.pop(tag)
        self._save(runs_dir)
        return removed

    def _save(self, runs_dir: Path) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / BASELINES_FILENAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Sort for stable diffs when this file is committed alongside runs.
        payload = json.dumps(dict(sorted(self.entries.items())), indent=2, sort_keys=True)
        tmp.write_text(payload)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


class AmbiguousRunRef(HistoryError):
    """A value matches both a baseline tag and a literal run id."""


def resolve_run_ref(
    runs_dir: Path,
    value: str,
    *,
    literal: bool = False,
    summaries: list[RunSummary] | None = None,
    baselines: BaselineRegistry | None = None,
) -> tuple[str, str | None]:
    """Resolve `value` to a concrete run id.

    Returns (run_id, banner) where `banner` is a one-line note when the
    resolution shadowed something the caller may want to know about (ambiguous
    tag/id) or None when the resolution was unambiguous.

    `literal=True` skips tag/reserved-name resolution and treats the value as
    a literal run id (still validated against `runs_dir`).
    """
    if literal:
        if not _run_exists(runs_dir, value):
            raise HistoryError(f"run {value!r} not found under {runs_dir}")
        return value, None

    baselines = baselines if baselines is not None else BaselineRegistry.load(runs_dir)
    tag_target = baselines.get(value)
    if tag_target is not None:
        try:
            target_exists = _run_exists(runs_dir, tag_target)
        except HistoryError as error:
            raise HistoryError(
                f"baseline tag {value!r} points to missing or invalid run {tag_target!r}"
            ) from error
        if not target_exists:
            raise HistoryError(
                f"baseline tag {value!r} points to missing or invalid run {tag_target!r}"
            )
    literal_exists = False
    try:
        literal_exists = _run_exists(runs_dir, value)
    except HistoryError:
        if tag_target is None:
            raise

    if value in RESERVED_REFS:
        summaries = summaries if summaries is not None else load_all_runs(runs_dir)[0]
        return _resolve_reserved(value, summaries), None

    if tag_target is not None and literal_exists and tag_target != value:
        banner = (
            f"value {value!r} matches both baseline tag and literal run dir; "
            f"resolved to tag → {tag_target}. Pass --literal to override."
        )
        return tag_target, banner
    if tag_target is not None:
        return tag_target, None
    if literal_exists:
        return value, None
    raise HistoryError(f"could not resolve {value!r} as run id, baseline tag, or reserved name")


def _resolve_reserved(value: str, summaries: list[RunSummary]) -> str:
    if not summaries:
        raise HistoryError(f"cannot resolve {value!r}: no runs found")
    if value == "latest":
        return summaries[0].run_id
    if value == "previous":
        if len(summaries) < 2:
            raise HistoryError(f"cannot resolve 'previous': only {len(summaries)} run available")
        return summaries[1].run_id
    raise HistoryError(f"unknown reserved ref {value!r}")


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


RegressionKind = Literal["score_drop", "pass_to_fail"]


class CompareRow(BaseModel):
    """Per (agent, task) comparison row between two runs."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    task_id: str
    score_a: float
    score_b: float
    delta_score: float
    latency_a: float | None
    latency_b: float | None
    delta_latency_s: float | None
    cost_a: float | None
    cost_b: float | None
    delta_cost_usd: float | None
    outcome_status_a: str
    outcome_status_b: str
    success_a: bool | None
    success_b: bool | None
    regression: RegressionKind | None


class ComparisonReport(BaseModel):
    """Side-by-side comparison of two runs."""

    model_config = ConfigDict(frozen=True)

    run_a: RunSummary
    run_b: RunSummary
    rows: list[CompareRow]
    tasks_only_in_a: list[str]
    tasks_only_in_b: list[str]
    agents_only_in_a: list[str]
    agents_only_in_b: list[str]
    harness_drift: str | None


def _index_results(
    results: Iterable[TaskRunResult],
) -> dict[tuple[str, str], TaskRunResult]:
    return {(r.agent_id, r.task_id): r for r in results}


# Regression display thresholds. Documented in design.md D6.
SCORE_REGRESSION_THRESHOLD = 0.10


def _classify_regression(
    score_a: float,
    score_b: float,
    success_a: bool | None,
    success_b: bool | None,
) -> RegressionKind | None:
    if success_a is True and success_b is False:
        return "pass_to_fail"
    if (score_a - score_b) > SCORE_REGRESSION_THRESHOLD:
        return "score_drop"
    return None


def _drift_banner(a: RunSummary, b: RunSummary) -> str | None:
    if a.harness_git_sha is None or b.harness_git_sha is None:
        return "harness drift unknown — at least one run did not record harness_git_sha"
    if a.harness_git_sha != b.harness_git_sha:
        return (
            f"harness drift: {a.run_id} at {a.harness_git_sha[:12]}, "
            f"{b.run_id} at {b.harness_git_sha[:12]}"
        )
    return None


def compare(runs_dir: Path, run_a: str, run_b: str) -> ComparisonReport:
    """Build a ComparisonReport for two runs identified by literal run ids."""
    a_dir = _run_dir_for_id(runs_dir, run_a)
    b_dir = _run_dir_for_id(runs_dir, run_b)
    if not (a_dir / "manifest.json").exists():
        raise HistoryError(f"run {run_a!r} not found under {runs_dir}")
    if not (b_dir / "manifest.json").exists():
        raise HistoryError(f"run {run_b!r} not found under {runs_dir}")

    manifest_a = load_manifest(a_dir)
    manifest_b = load_manifest(b_dir)
    results_a = load_run_results(runs_dir, run_a)
    results_b = load_run_results(runs_dir, run_b)
    summary_a = _summary_from_manifest(manifest_a, results_a)
    summary_b = _summary_from_manifest(manifest_b, results_b)

    idx_a = _index_results(results_a)
    idx_b = _index_results(results_b)
    shared_keys = sorted(set(idx_a) & set(idx_b))

    rows: list[CompareRow] = []
    for agent, task in shared_keys:
        a = idx_a[(agent, task)]
        b = idx_b[(agent, task)]
        latency_delta = (
            b.latency_s - a.latency_s
            if a.latency_s is not None and b.latency_s is not None
            else None
        )
        cost_delta = (
            b.cost_usd - a.cost_usd if a.cost_usd is not None and b.cost_usd is not None else None
        )
        rows.append(
            CompareRow(
                agent_id=agent,
                task_id=task,
                score_a=a.score,
                score_b=b.score,
                delta_score=b.score - a.score,
                latency_a=a.latency_s,
                latency_b=b.latency_s,
                delta_latency_s=latency_delta,
                cost_a=a.cost_usd,
                cost_b=b.cost_usd,
                delta_cost_usd=cost_delta,
                outcome_status_a=a.outcome_status,
                outcome_status_b=b.outcome_status,
                success_a=a.success,
                success_b=b.success,
                regression=_classify_regression(a.score, b.score, a.success, b.success),
            )
        )

    tasks_a = set(summary_a.task_ids)
    tasks_b = set(summary_b.task_ids)
    agents_a = set(summary_a.agent_ids)
    agents_b = set(summary_b.agent_ids)

    return ComparisonReport(
        run_a=summary_a,
        run_b=summary_b,
        rows=rows,
        tasks_only_in_a=sorted(tasks_a - tasks_b),
        tasks_only_in_b=sorted(tasks_b - tasks_a),
        agents_only_in_a=sorted(agents_a - agents_b),
        agents_only_in_b=sorted(agents_b - agents_a),
        harness_drift=_drift_banner(summary_a, summary_b),
    )
