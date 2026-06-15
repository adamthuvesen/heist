from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import heist.history as history_module
from heist.cli import app
from heist.runner import run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


def _seed_live_run(tmp_path: Path, run_id: str = "live-1") -> Path:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, _ = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id=run_id,
    )
    return Path(manifest.run_dir)


def test_replay_all_happy_path(tmp_path: Path) -> None:
    _seed_live_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "live-1",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "replay-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Replay" in result.output
    assert "replay-1" in result.output
    replay_dir = tmp_path / "runs" / "replay-1"
    assert (replay_dir / "manifest.json").exists()
    assert (replay_dir / "results.jsonl").exists()
    assert (replay_dir / "report.html").exists()


def test_replay_subset_by_task(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "a")
    write_marker_task(tmp_path, "b")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="live-subset",
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "live-subset",
            "--task",
            "a",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "replay-subset",
        ],
    )
    assert result.exit_code == 0, result.output
    from heist.runner import load_results

    rows = load_results(tmp_path / "runs" / "replay-subset")
    assert {row.task_id for row in rows} == {"a"}


def test_replay_rejects_unknown_agent(tmp_path: Path) -> None:
    _seed_live_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "live-1",
            "--agent",
            "ghost",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "no rows for agent" in result.output
    # New run dir must not exist on rejection.
    assert not (tmp_path / "runs" / "replay-of-live-1").exists()


def test_replay_rejects_unknown_task(tmp_path: Path) -> None:
    _seed_live_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "live-1",
            "--task",
            "ghost-task",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "no rows for task" in result.output


def test_replay_rejects_replay_of_replay(tmp_path: Path) -> None:
    live_dir = _seed_live_run(tmp_path)
    # First replay to create a kind="replay" source.
    first = runner.invoke(
        app,
        [
            "runs",
            "replay",
            live_dir.name,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "first-replay",
        ],
    )
    assert first.exit_code == 0, first.output

    # Now try to replay the replay.
    second = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "first-replay",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "should-fail",
        ],
    )
    assert second.exit_code == 1
    assert "is itself" in second.output
    assert "replay" in second.output
    assert not (tmp_path / "runs" / "should-fail").exists()


def test_replay_resolves_baseline_tag(tmp_path: Path) -> None:
    _seed_live_run(tmp_path)
    from heist.history import BaselineRegistry

    BaselineRegistry.load(tmp_path / "runs").set(tmp_path / "runs", "live-1", "v1")
    result = runner.invoke(
        app,
        [
            "runs",
            "replay",
            "v1",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "replay-via-tag",
        ],
    )
    assert result.exit_code == 0, result.output
    from heist.runner import load_manifest

    manifest = load_manifest(tmp_path / "runs" / "replay-via-tag")
    assert manifest.source_run_id == "live-1"
    assert manifest.kind == "replay"
