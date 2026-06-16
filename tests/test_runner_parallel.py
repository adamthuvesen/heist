from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from heist.models import AgentSpec, TaskDefinition, TaskRunResult
from heist.runner import _resolve_provider_caps, run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import write_marker_task


def _slow_agent(provider: str, mode: str = "pass") -> AgentSpec:
    script = Path(__file__).parent / "fixtures" / "fake_agent.py"
    return AgentSpec(
        id=f"{provider}-{mode}",
        label=f"{provider} {mode}",
        provider=provider,
        model_id=f"{provider}-model",
        command=[sys.executable, str(script), mode],
    )


class _RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None:
        with self._lock:
            self.events.append(("start", agent.id))

    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None:
        with self._lock:
            self.events.append(("finish", agent.id))


def test_resolve_provider_caps_defaults_to_global_jobs() -> None:
    agents = [_slow_agent("claude"), _slow_agent("cursor"), _slow_agent("codex")]
    caps = _resolve_provider_caps(agents, jobs=4, provider_jobs=None)
    assert caps == {"claude": 4, "cursor": 4, "codex": 4}


def test_resolve_provider_caps_clipped_to_global() -> None:
    agents = [_slow_agent("claude"), _slow_agent("cursor")]
    caps = _resolve_provider_caps(agents, jobs=3, provider_jobs={"claude": 10, "cursor": 1})
    assert caps == {"claude": 3, "cursor": 1}


def test_resolve_provider_caps_rejects_zero() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_provider_caps([_slow_agent("claude")], jobs=2, provider_jobs={"claude": 0})


def test_run_benchmark_emits_start_before_finish_per_agent(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    reporter = _RecordingReporter()

    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[_slow_agent("alpha"), _slow_agent("beta")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="reporter-run",
        jobs=2,
        reporter=reporter,
    )

    assert manifest.run_id == "reporter-run"
    # Real invariant: every agent's `finish` event comes after its `start`.
    # Sorting the events list would hide an out-of-order pair.
    for agent_id in ["alpha-pass", "beta-pass"]:
        start_idx = reporter.events.index(("start", agent_id))
        finish_idx = reporter.events.index(("finish", agent_id))
        assert start_idx < finish_idx, reporter.events
    assert all(result.success is True for result in results)


def test_run_benchmark_provider_cap_serializes_same_provider(tmp_path: Path) -> None:
    # Two slow agents on the same provider with cap=1 should run sequentially.
    # Verify by counting *concurrently running* jobs from the same provider —
    # a shared counter under a lock catches over-concurrency without depending
    # on wall-clock overlap (which is flaky under CI load).
    write_marker_task(tmp_path, "marker")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    running_by_provider: dict[str, int] = {"claude": 0}
    peak_by_provider: dict[str, int] = {"claude": 0}
    counter_lock = threading.Lock()

    class _ConcurrencyReporter:
        def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None:
            with counter_lock:
                running_by_provider[agent.provider] += 1
                peak_by_provider[agent.provider] = max(
                    peak_by_provider[agent.provider],
                    running_by_provider[agent.provider],
                )

        def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None:
            with counter_lock:
                running_by_provider[agent.provider] -= 1

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[
            _slow_agent("claude", "delayed_pass"),
            _slow_agent("claude", "pass"),
        ],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="cap-run",
        jobs=4,
        provider_jobs={"claude": 1},
        reporter=_ConcurrencyReporter(),
    )

    # Both agents must actually run (rules out a no-op silently satisfying the cap).
    assert len(results) >= 2
    # Provider cap=1 means the in-flight count for that provider never exceeds 1.
    assert peak_by_provider["claude"] == 1, peak_by_provider


def test_run_benchmark_fail_fast_skips_pending_jobs(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "marker-a")
    write_marker_task(tmp_path, "marker-b")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    # Break grader for marker-a (alphabetically first) so its result is errored,
    # triggering fail-fast on the very first job.
    bad_grader = tmp_path / "tasks" / "smoke" / "marker-a" / "hidden" / "grader.py"
    bad_grader.write_text("raise RuntimeError('boom')")

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[_slow_agent("alpha", "pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="fail-fast-run",
        jobs=1,
        fail_fast=True,
    )

    # Exactly one row: marker-a (errored). marker-b never ran.
    assert len(results) == 1, results
    assert results[0].task_id == "marker-a"
    assert results[0].outcome_status == "errored"
    marker_b_workspace = (
        tmp_path / "runs" / "fail-fast-run" / "workspaces" / "alpha-pass" / "marker-b"
    )
    assert not marker_b_workspace.exists(), "marker-b workspace was created despite fail-fast"


def test_run_benchmark_fail_fast_cancels_parallel_pending(tmp_path: Path) -> None:
    # The parallel (jobs>1) abort_event path: when one job errors, in-flight
    # workers must be cancelled (pgid_registry kill) and pending submissions
    # must raise _AbortedJob before starting.
    for task_id in ["marker-a", "marker-b", "marker-c", "marker-d"]:
        write_marker_task(tmp_path, task_id)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    # marker-a fails fast: its grader raises immediately and triggers the
    # fail-fast path. marker-b/c/d each have a slow grader so that when
    # fail-fast triggers, marker-b is still mid-grade (in-flight kill path)
    # and marker-c/d are still queued (_AbortedJob path). Without this delay
    # all four can complete in a single scheduler tick on a fast machine —
    # the L15 flake.
    bad_grader = tmp_path / "tasks" / "smoke" / "marker-a" / "hidden" / "grader.py"
    bad_grader.write_text("raise RuntimeError('boom')")
    for task_id in ["marker-b", "marker-c", "marker-d"]:
        slow = tmp_path / "tasks" / "smoke" / task_id / "hidden" / "grader.py"
        slow.write_text(
            "import time, sys\n"
            "time.sleep(0.8)\n"
            'print(\'{"score":1.0,"passed":true,"checks":[]}\')\n'
        )

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[_slow_agent("alpha", "pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="fail-fast-parallel-run",
        jobs=2,
        fail_fast=True,
    )

    # At least marker-a errored. fail-fast must prevent at least one of the
    # remaining tasks from completing (typically marker-d, the last submitted).
    errored = [r for r in results if r.outcome_status == "errored"]
    assert errored, results
    assert len(results) < 4, (
        f"fail-fast didn't skip any pending parallel jobs (got {len(results)}/4)"
    )


def test_run_benchmark_fail_fast_drops_inflight_killed_jobs(tmp_path: Path) -> None:
    # H3: a job killed mid-flight by the fail-fast SIGTERM must be DROPPED, not
    # recorded as a spurious 'errored' row that blames the agent for the
    # harness's own interruption. marker-a errors instantly and triggers the
    # abort; marker-b's grader sleeps, so it is still mid-grade when the SIGTERM
    # lands and its process group is killed.
    write_marker_task(tmp_path, "marker-a")
    write_marker_task(tmp_path, "marker-b")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    (tmp_path / "tasks" / "smoke" / "marker-a" / "hidden" / "grader.py").write_text(
        "raise RuntimeError('boom')"
    )
    (tmp_path / "tasks" / "smoke" / "marker-b" / "hidden" / "grader.py").write_text(
        'import time\ntime.sleep(0.8)\nprint(\'{"score":1.0,"passed":true,"checks":[]}\')\n'
    )

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[_slow_agent("alpha", "pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="fail-fast-inflight-run",
        jobs=2,
        fail_fast=True,
    )

    errored_ids = {r.task_id for r in results if r.outcome_status == "errored"}
    # The genuine trigger is recorded...
    assert "marker-a" in errored_ids, results
    # ...but the interrupted-mid-flight job is not blamed as a failure.
    assert "marker-b" not in errored_ids, (
        f"interrupted job recorded as errored instead of dropped: {results}"
    )
