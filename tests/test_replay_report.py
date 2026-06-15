from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import heist.history as history_module
from heist.cli import app
from heist.reporting import render_html, write_report
from heist.runner import run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task
from tests.fixtures.runs import make_result, write_synthetic_run

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


def test_kind_badge_absent_on_live_render(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "live", results=[make_result(run_id="live")])
    results = history_module.load_run_results(tmp_path, "live")
    html = render_html(results)
    assert '<div class="kind-badge">' not in html
    assert "{{KIND_BADGE}}" not in html


def test_kind_badge_renders_when_replay_source_supplied(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "replay", results=[make_result(run_id="replay")])
    results = history_module.load_run_results(tmp_path, "replay")
    html = render_html(results, replay_source_run_id="origin-live-1")
    assert '<div class="kind-badge">' in html
    assert "Replay" in html
    assert "origin-live-1" in html
    assert "agents not measured" in html


def test_write_report_persists_kind_badge(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "rp", results=[make_result(run_id="rp")])
    results = history_module.load_run_results(tmp_path, "rp")
    write_report(tmp_path / "rp", results, replay_source_run_id="my-source")
    rendered = (tmp_path / "rp" / "report.html").read_text()
    assert '<div class="kind-badge">' in rendered
    assert "my-source" in rendered


def test_report_cli_picks_up_replay_kind_from_manifest(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    # Live run as source.
    manifest, _ = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="live",
    )
    live_dir = Path(manifest.run_dir)

    # Replay through the CLI (uses SnapshotExecutor).
    result = runner.invoke(
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
            "rp",
        ],
    )
    assert result.exit_code == 0, result.output

    # Re-render the replay's report via `heist report`: it must read manifest
    # and reapply the badge even when no badge flag is passed explicitly.
    rerender = runner.invoke(
        app,
        ["report", "--run", str(tmp_path / "runs" / "rp")],
    )
    assert rerender.exit_code == 0, rerender.output
    html = (tmp_path / "runs" / "rp" / "report.html").read_text()
    assert '<div class="kind-badge">' in html
    assert "live" in html  # source_run_id appears in the badge
