from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import heist.history as history_module
from heist.cli import app
from tests.fixtures.runs import (
    make_result,
    write_corrupt_run,
    write_synthetic_run,
    write_two_runs,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_results_cache() -> None:
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


def _invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_runs_list_empty(tmp_path: Path) -> None:
    code, output = _invoke("runs", "list", "--runs-dir", str(tmp_path))
    assert code == 0
    assert "No runs found" in output


def test_runs_list_renders_both_runs(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    code, output = _invoke("runs", "list", "--runs-dir", str(tmp_path))
    assert code == 0, output
    assert "run-a" in output
    assert "run-b" in output
    # Newest (run-b) renders first.
    assert output.index("run-b") < output.index("run-a")


def test_runs_list_flags_corrupt_run(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    write_corrupt_run(tmp_path, "broken")
    code, output = _invoke("runs", "list", "--runs-dir", str(tmp_path), "--include-corrupt")
    assert code == 0
    assert "ok" in output
    assert "broken" in output
    assert "unreadable" in output


def test_runs_compare_outputs_delta_table(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    code, output = _invoke("runs", "compare", "run-a", "run-b", "--runs-dir", str(tmp_path))
    assert code == 0, output
    assert "Agent" in output
    # Δ score must include both signs/headers we render.
    assert "score" in output
    assert "drift" in output  # harness drift banner from differing SHAs in fixture
    # Regression flag for the −0.3 score drop.
    assert "score drop" in output or "drop" in output


def test_runs_compare_with_baseline_flag(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    from heist.history import BaselineRegistry

    BaselineRegistry.load(tmp_path).set(tmp_path, "run-a", "v1")
    code, output = _invoke(
        "runs",
        "compare",
        "run-b",
        "--baseline",
        "v1",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 0, output
    assert "v1 → run-a" in output


def test_runs_compare_rejects_baseline_with_two_positional(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    code, output = _invoke(
        "runs",
        "compare",
        "run-a",
        "run-b",
        "--baseline",
        "v1",
        "--runs-dir",
        str(tmp_path),
    )
    assert code != 0
    assert "exactly one positional" in output


def test_runs_compare_requires_two_refs(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    code, output = _invoke("runs", "compare", "run-a", "--runs-dir", str(tmp_path))
    assert code != 0
    assert "two run references" in output


def test_runs_compare_unknown_ref_exits_nonzero(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    code, output = _invoke("runs", "compare", "only", "ghost", "--runs-dir", str(tmp_path))
    assert code == 1
    assert "could not resolve" in output


def test_runs_history_renders_chronologically(tmp_path: Path) -> None:
    older, newer = write_two_runs(tmp_path, agent_id="claude", task_id="auth")
    code, output = _invoke(
        "runs",
        "history",
        "--agent",
        "claude",
        "--task",
        "auth",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 0, output
    assert older in output
    assert newer in output
    assert output.index(older) < output.index(newer)
    assert "n=2" in output


def test_runs_history_unknown_agent_exits(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "r1", results=[make_result(run_id="r1")])
    code, output = _invoke(
        "runs",
        "history",
        "--agent",
        "ghost-agent",
        "--task",
        "marker",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 1
    assert "Unknown agent" in output


def test_runs_history_unknown_task_exits(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "r1", results=[make_result(run_id="r1")])
    code, output = _invoke(
        "runs",
        "history",
        "--agent",
        "fake-pass",
        "--task",
        "ghost-task",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 1
    assert "Unknown task" in output


def test_runs_history_no_history_message(tmp_path: Path) -> None:
    write_synthetic_run(
        tmp_path,
        "r1",
        results=[make_result(run_id="r1", agent_id="a", task_id="t")],
    )
    write_synthetic_run(
        tmp_path,
        "r2",
        results=[make_result(run_id="r2", agent_id="b", task_id="u")],
    )
    code, output = _invoke(
        "runs",
        "history",
        "--agent",
        "a",
        "--task",
        "u",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 0
    assert "No history" in output


def test_baseline_set_list_unset_roundtrip(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    code, output = _invoke("runs", "baseline", "list", "--runs-dir", str(tmp_path))
    assert code == 0
    assert "No baseline tags" in output

    code, output = _invoke("runs", "baseline", "set", "ok", "v1", "--runs-dir", str(tmp_path))
    assert code == 0
    assert "Set baseline" in output

    code, output = _invoke("runs", "baseline", "list", "--runs-dir", str(tmp_path))
    assert code == 0
    assert "v1" in output and "ok" in output

    code, output = _invoke("runs", "baseline", "unset", "v1", "--runs-dir", str(tmp_path))
    assert code == 0
    assert "Removed baseline" in output


def test_baseline_set_reserved_name_rejected(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    code, output = _invoke("runs", "baseline", "set", "ok", "latest", "--runs-dir", str(tmp_path))
    assert code == 1
    assert "reserved" in output


def test_baseline_set_missing_run_rejected(tmp_path: Path) -> None:
    code, output = _invoke("runs", "baseline", "set", "ghost", "v1", "--runs-dir", str(tmp_path))
    assert code == 1
    assert "not found" in output


def test_baseline_unset_unknown_tag_rejected(tmp_path: Path) -> None:
    code, output = _invoke("runs", "baseline", "unset", "missing", "--runs-dir", str(tmp_path))
    assert code == 1
    assert "not defined" in output
