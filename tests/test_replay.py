from __future__ import annotations

from pathlib import Path

import pytest

import heist.history as history_module
from heist.models import TaskRunResult
from heist.replay import (
    ReplayOfReplayError,
    SnapshotExecutor,
)
from heist.runner import run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task
from tests.fixtures.runs import make_result, write_synthetic_run


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_snapshot_executor_refuses_replay_of_replay(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    write_synthetic_run(
        runs_dir,
        "live-source",
        results=[make_result(run_id="live-source")],
    )
    write_synthetic_run(
        runs_dir,
        "first-replay",
        results=[make_result(run_id="first-replay")],
        kind="replay",
        source_run_id="live-source",
    )
    with pytest.raises(ReplayOfReplayError, match="live-source"):
        SnapshotExecutor(runs_dir / "first-replay")


def test_snapshot_executor_loads_known_pairs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    write_synthetic_run(
        runs_dir,
        "src",
        results=[
            make_result(run_id="src", agent_id="a", task_id="t1"),
            make_result(run_id="src", agent_id="b", task_id="t2"),
        ],
    )
    executor = SnapshotExecutor(runs_dir / "src")
    assert executor.known_pairs() == {("a", "t1"), ("b", "t2")}
    assert executor.known_agents() == {"a", "b"}
    assert executor.known_tasks() == {"t1", "t2"}


# ---------------------------------------------------------------------------
# End-to-end replay (live run, then replay through run_benchmark)
# ---------------------------------------------------------------------------


def _run_live(
    tmp_path: Path, run_id: str, *, mode: str = "pass"
) -> tuple[Path, list[TaskRunResult]]:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent(mode)],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id=run_id,
    )
    return Path(manifest.run_dir), results


def test_replay_preserves_pass_outcome_and_marks_manifest(tmp_path: Path) -> None:
    live_dir, live_results = _run_live(tmp_path, "live-pass")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    replay_manifest, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-pass",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-pass",
    )

    assert replay_manifest.kind == "replay"
    assert replay_manifest.source_run_id == "live-pass"
    [replay_row] = replay_results
    [live_row] = live_results
    # Grader produces the same score on the replayed workspace.
    assert replay_row.success is True
    assert replay_row.score == live_row.score
    # Latency / usage / cost faithfully copied from source.
    assert replay_row.latency_s == live_row.latency_s
    assert replay_row.tokens_in == live_row.tokens_in
    assert replay_row.tokens_out == live_row.tokens_out
    # No agent CLI was invoked — the captured stdout was copied through.
    assert Path(replay_row.stdout_path).read_text() == Path(live_row.stdout_path).read_text()
    # Diff artefact mirrors source's diff bytes.
    assert Path(replay_row.diff_path).read_bytes() == Path(live_row.diff_path).read_bytes()


def test_replay_preserves_fail_outcome(tmp_path: Path) -> None:
    live_dir, live_results = _run_live(tmp_path, "live-fail", mode="fail")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("fail")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-fail",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-fail",
    )

    [replay_row] = replay_results
    [live_row] = live_results
    # Grader on the same final workspace reproduces the same failure.
    assert replay_row.success is False
    assert replay_row.score == live_row.score


def test_replay_preserves_timeout_without_running_grader(tmp_path: Path) -> None:
    live_dir, live_results = _run_live(tmp_path, "live-timeout", mode="slow_usage")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    # The live run must have actually timed out for this test to be meaningful.
    [live_row] = live_results
    # `slow_usage` fixture may or may not time out depending on the env's
    # timeout setup; the harness uses timeout_s=5 above, while the fixture
    # sleeps. Skip with a clear message if it didn't.
    if not live_row.timed_out:
        pytest.skip("slow_usage fixture did not time out in this environment")

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("slow_usage")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-timeout",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-timeout",
    )
    [replay_row] = replay_results
    assert replay_row.timed_out is True
    assert replay_row.outcome_status == "errored"
    assert replay_row.success is None


def test_replay_propagates_missing_env_terminal_outcome(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    needy = fake_agent("pass").model_copy(
        update={"id": "fake-needy", "required_env": ["HEIST_NEVER_SET_XYZ"]}
    )
    manifest, _ = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[needy],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="live-env",
    )
    live_dir = Path(manifest.run_dir)

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[needy],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-env",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-env",
    )
    [replay_row] = replay_results
    assert replay_row.outcome_status == "errored"
    assert "HEIST_NEVER_SET_XYZ" in (replay_row.error or "")


# ---------------------------------------------------------------------------
# Source-missing failure modes
# ---------------------------------------------------------------------------


def test_replay_missing_source_workspace_yields_errored_result(tmp_path: Path) -> None:
    live_dir, _ = _run_live(tmp_path, "live-pass-2")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    # Delete the captured workspace; results.jsonl still references the
    # original path, so the snapshot loader will fail to copy.
    import shutil as shutil_mod

    shutil_mod.rmtree(live_dir / "workspaces")

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-missing-ws",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-pass-2",
        retry=0,
    )
    [replay_row] = replay_results
    assert replay_row.outcome_status == "errored"
    assert "source workspace missing" in (replay_row.error or "")


def test_replay_source_workspace_file_yields_errored_result(tmp_path: Path) -> None:
    live_dir, _ = _run_live(tmp_path, "live-pass-file")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    source_ws = live_dir / "workspaces" / "fake-pass" / "marker"
    import shutil as shutil_mod

    shutil_mod.rmtree(source_ws)
    source_ws.write_text("not a directory")

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-file-ws",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-pass-file",
        retry=0,
    )
    [replay_row] = replay_results
    assert replay_row.outcome_status == "errored"
    assert "source workspace is not a directory" in (replay_row.error or "")


def test_replay_missing_source_row_yields_errored_result(tmp_path: Path) -> None:
    live_dir, _ = _run_live(tmp_path, "live-pass-3")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    # Truncate results.jsonl so the (agent, task) row is gone, but workspace
    # is still present. SnapshotExecutor.run() must raise ReplaySourceMissing,
    # which the runner converts to an errored result.
    (live_dir / "results.jsonl").write_text("")

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-missing-row",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-pass-3",
        retry=0,
    )
    [replay_row] = replay_results
    assert replay_row.outcome_status == "errored"
    assert "source row missing" in (replay_row.error or "")


# ---------------------------------------------------------------------------
# Cost re-derivation
# ---------------------------------------------------------------------------


def test_replay_cost_fields_match_source_when_pricing_unchanged(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    # `reported` fake agent emits a reported cost AND token usage we can price.
    manifest, live_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="live-cost",
    )
    live_dir = Path(manifest.run_dir)

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-cost",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-cost",
    )
    [replay_row] = replay_results
    [live_row] = live_results
    assert replay_row.reported_session_cost_usd == live_row.reported_session_cost_usd
    assert replay_row.reconstructed_per_task_cost_usd == live_row.reconstructed_per_task_cost_usd
    assert replay_row.cost_usd == live_row.cost_usd
    assert replay_row.cost_source == live_row.cost_source
    assert replay_row.cost_provenance == live_row.cost_provenance


def test_replay_reconstructed_cost_reflects_pricing_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, live_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="live-pricing",
    )
    live_dir = Path(manifest.run_dir)
    [live_row] = live_results
    assert live_row.reconstructed_per_task_cost_usd is not None

    # Triple input + output pricing for gpt-5.5 between source and replay.
    import heist.usage as usage_module

    new_pricing = dict(usage_module.PRICING_PER_MILLION)
    new_pricing["gpt-5.5"] = (15.0, 90.0, 0.5, 0.0)
    monkeypatch.setattr(usage_module, "PRICING_PER_MILLION", new_pricing)

    _, replay_results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="replay-pricing",
        executor=SnapshotExecutor(live_dir),
        kind="replay",
        source_run_id="live-pricing",
    )
    [replay_row] = replay_results
    # Token counts come from source; reconstructed cost reflects new pricing.
    assert replay_row.tokens_in == live_row.tokens_in
    assert replay_row.tokens_out == live_row.tokens_out
    assert replay_row.reported_session_cost_usd == live_row.reported_session_cost_usd
    assert replay_row.reconstructed_per_task_cost_usd is not None
    assert replay_row.reconstructed_per_task_cost_usd == pytest.approx(
        live_row.reconstructed_per_task_cost_usd * 3
    )


def test_overlay_workspace_skips_dotgit(tmp_path: Path) -> None:
    # Manual SnapshotExecutor unit, not via run_benchmark.
    runs_dir = tmp_path / "runs"
    write_synthetic_run(runs_dir, "src", results=[make_result(run_id="src")])
    # Fabricate a captured workspace next to the synthetic run.
    source_ws = runs_dir / "src" / "workspaces" / "fake-pass" / "marker"
    source_ws.mkdir(parents=True)
    (source_ws / ".git").mkdir()
    (source_ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (source_ws / "answer.txt").write_text("yes\n")

    dest = tmp_path / "dest"
    from heist.replay import _copy_source_tree

    _copy_source_tree(source_ws, dest)
    assert (dest / "answer.txt").read_text() == "yes\n"
    # `.git` must not leak from source — _baseline_workspace builds its own.
    assert not (dest / ".git").exists()
