from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from heist.runner import _capture_harness_sha, run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task


def test_capture_harness_sha_returns_sha_in_git_checkout() -> None:
    # The test suite itself runs inside the heist checkout, so capture must
    # succeed here. Asserting a 40-char hex shape avoids pinning to a specific
    # commit that drifts as the branch advances.
    sha = _capture_harness_sha()
    assert sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}", sha), sha


def test_capture_harness_sha_returns_none_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git not on PATH")

    monkeypatch.setattr("heist.runner.subprocess.run", fake_run)
    assert _capture_harness_sha() is None


def test_capture_harness_sha_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)

    monkeypatch.setattr("heist.runner.subprocess.run", fake_run)
    assert _capture_harness_sha() is None


def test_capture_harness_sha_returns_none_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(args[0]) if args else [],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository\n",
        )

    monkeypatch.setattr("heist.runner.subprocess.run", fake_run)
    assert _capture_harness_sha() is None


def test_run_benchmark_records_harness_sha_in_manifest(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, _ = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="sha-run",
    )
    # The harness invokes git for its own checkout, not the tmp benchmark dir,
    # so the captured SHA reflects the real heist tree.
    assert manifest.harness_git_sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.harness_git_sha)
