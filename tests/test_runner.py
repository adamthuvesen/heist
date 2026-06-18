from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from heist.agents import DEFAULT_AGENTS
from heist.models import AgentSpec
from heist.runner import (
    _ordered_jobs,
    execute_agent,
    load_manifest,
    load_results,
    make_run_id,
    regrade_run,
    run_benchmark,
    validate_run_id,
)
from heist.subprocess_utils import SubprocessResult
from heist.tasks import load_tasks, select_tasks
from tests.fixtures.marker import FAKE_AGENT_SCRIPT, break_grader, fake_agent, write_marker_task
from tests.fixtures.runs import make_result, write_synthetic_run


def test_run_benchmark_with_fake_pass_and_fail_agents(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass"), fake_agent("fail")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="test-run",
        jobs=1,
    )

    assert manifest.run_id == "test-run"
    by_agent = {result.agent_id: result for result in results}
    assert by_agent["fake-pass"].success is True
    assert by_agent["fake-fail"].success is False
    assert Path(by_agent["fake-pass"].diff_path).read_text()
    assert len(load_results(Path(manifest.run_dir))) == 2
    loaded_manifest = load_manifest(Path(manifest.run_dir))
    assert loaded_manifest.completed_at is not None
    assert loaded_manifest.duration_s is not None
    # > 0 (not >= 0) confirms wall-clock was observed; duration_s is
    # non-negative by construction so >= 0 added no signal.
    assert loaded_manifest.duration_s > 0
    assert loaded_manifest.status == "completed"


def test_run_benchmark_parallel_jobs_preserve_selection_order(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("delayed_pass"), fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="parallel-run",
        jobs=2,
    )

    assert [result.agent_id for result in results] == [
        "fake-delayed_pass",
        "fake-pass",
    ]
    assert [result.success for result in results] == [True, True]
    assert [result.agent_id for result in load_results(Path(manifest.run_dir))] == [
        "fake-delayed_pass",
        "fake-pass",
    ]


def test_ordered_jobs_iterates_agents_outer_tasks_inner(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "a")
    write_marker_task(tmp_path, "b")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    agents = [fake_agent("pass"), fake_agent("fail")]

    ordered = _ordered_jobs(agents, tasks)

    assert [(agent.id, task.id) for _, agent, task in ordered] == [
        ("fake-pass", "a"),
        ("fake-pass", "b"),
        ("fake-fail", "a"),
        ("fake-fail", "b"),
    ]
    assert [index for index, _, _ in ordered] == [0, 1, 2, 3]


def test_run_benchmark_rejects_invalid_jobs(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    with pytest.raises(ValueError, match="jobs must be at least 1"):
        run_benchmark(
            repo_root=tmp_path,
            suite="smoke",
            agents=[fake_agent("pass")],
            tasks=tasks,
            runs_dir=tmp_path / "runs",
            timeout_s=5,
            run_id="invalid-jobs-run",
            jobs=0,
        )


def test_run_benchmark_prefers_reported_cost(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="reported-run",
    )

    [result] = results
    assert result.success is True
    assert result.reported_session_cost_usd == 1.23
    assert result.reconstructed_per_task_cost_usd == 0.0011
    assert result.cost_usd == 1.23
    assert result.cost_source == "reported"
    assert result.cost_provenance == "reconciled"


def test_run_benchmark_records_cost_for_non_timeout_errors(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    break_grader(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="error-run",
    )

    [result] = results
    # Errored rows must keep `success` as None (not False): False reads as
    # "graded and failed", which feeds different headlines/summaries.
    assert result.outcome_status == "errored"
    assert result.success is None
    assert result.score == 0.0
    assert result.timed_out is False
    assert result.cost_usd == 1.23
    assert result.cost_source == "reported"
    assert "grader exploded" in (result.error or ""), result.error


def test_regrade_run_rejects_rows_for_missing_current_tasks(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "marker")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    run_dir = write_synthetic_run(
        tmp_path / "runs",
        "stale-run",
        results=[make_result(run_id="stale-run", task_id="ghost-task")],
    )

    with pytest.raises(ValueError, match="references task 'ghost-task'"):
        regrade_run(run_dir, tasks)


def test_regrade_run_preserves_rows_with_missing_workspace(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "marker")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    # An errored row whose workspace was never created (default workspace_path
    # points at a nonexistent /tmp path).
    errored = make_result(
        run_id="env-fail",
        task_id="marker",
        outcome_status="errored",
        success=None,
        score=0.0,
    ).model_copy(update={"error": "agent 'fake' requires env vars: OPENROUTER_API_KEY"})
    run_dir = write_synthetic_run(tmp_path / "runs", "env-fail", results=[errored])

    [regraded] = regrade_run(run_dir, tasks)

    # The original failure is carried forward, not silently rewritten to graded.
    assert regraded.outcome_status == "errored"
    assert regraded.success is None
    assert regraded.score == 0.0
    assert regraded.error == "agent 'fake' requires env vars: OPENROUTER_API_KEY"


def test_run_benchmark_records_timeout_as_errored(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("slow_usage", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=1,
        run_id="timeout-run",
    )

    [result] = results
    assert result.outcome_status == "errored"
    assert result.success is None
    assert result.timed_out is True
    assert result.tokens_in == 100
    assert result.tokens_out == 20
    assert result.cost_usd == 0.0011
    assert result.cost_source == "reconstructed"


def test_run_benchmark_records_nonzero_agent_exit_as_errored(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("exit_nonzero")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="nonzero-run",
    )

    [result] = results
    assert result.outcome_status == "errored"
    assert result.success is None
    assert result.score == 0.0
    assert result.agent_exit_code == 2
    assert result.error == "agent exited with code 2"


def test_run_benchmark_reports_signal_death_as_killed_by_signal(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("signal_death")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="signal-run",
    )

    [result] = results
    assert result.outcome_status == "errored"
    assert result.timed_out is False
    # Negative returncode (-15) must be reported as a signal kill, not "code -15".
    assert result.error == "agent killed by signal 15"


def test_run_benchmark_kills_sigterm_ignoring_agent(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    start = time.monotonic()
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("ignore_sigterm")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=1,
        run_id="sigterm-ignore-run",
    )
    elapsed = time.monotonic() - start

    [result] = results
    assert result.timed_out is True
    assert result.outcome_status == "errored"
    # The agent's own sleep is 30s; if escalation to SIGKILL works it dies within
    # the timeout + grace + reap window, well under that.
    assert elapsed < 20


def test_write_diff_failure_produces_errored_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess as subprocess_module

    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    real_run = subprocess_module.run

    def fake_run(args, *a, **kw):  # type: ignore[no-untyped-def]
        if isinstance(args, list) and "diff" in args and args[0] == "git":
            # The runner now passes git config-scrub flags ahead of `diff`, so
            # match on membership rather than a fixed prefix. Return bytes
            # because runner uses capture_output without text=True for diff.
            return subprocess_module.CompletedProcess(
                args=args,
                returncode=128,
                stdout=b"",
                stderr=b"fatal: bad object HEAD\n",
            )
        return real_run(args, *a, **kw)

    monkeypatch.setattr("heist.runner.subprocess.run", fake_run)

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="diff-fail-run",
    )

    [result] = results
    assert result.outcome_status == "errored"
    assert "git diff failed" in (result.error or "")


def test_missing_required_env_produces_errored_row_not_abort(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "a")
    write_marker_task(tmp_path, "b")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    needy = fake_agent("pass")
    needy = needy.model_copy(update={"id": "fake-needy", "required_env": ["HEIST_NEVER_SET_XYZ"]})
    ok = fake_agent("pass")

    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[needy, ok],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="missing-env-run",
    )

    by_agent: dict[str, list] = {}
    for result in results:
        by_agent.setdefault(result.agent_id, []).append(result)

    assert {r.outcome_status for r in by_agent["fake-needy"]} == {"errored"}
    assert all("HEIST_NEVER_SET_XYZ" in (r.error or "") for r in by_agent["fake-needy"])
    assert all(r.success is True for r in by_agent["fake-pass"])


def test_make_run_id_includes_subsecond_and_random_suffix() -> None:
    run_id = make_run_id()
    # YYYYMMDD-HHMMSS-micros-hex4
    parts = run_id.split("-")
    assert len(parts) == 4, run_id
    date, time_part, micros, suffix = parts
    assert len(date) == 8 and date.isdigit(), run_id
    assert len(time_part) == 6 and time_part.isdigit(), run_id
    assert len(micros) == 6 and micros.isdigit(), run_id
    assert len(suffix) == 4 and all(c in "0123456789abcdef" for c in suffix), run_id


@pytest.mark.parametrize("run_id", ["../outside", "/tmp/outside", "nested/run"])
def test_validate_run_id_rejects_path_like_values(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid run id"):
        validate_run_id(run_id)


def test_run_benchmark_rejects_path_like_run_id(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    with pytest.raises(ValueError, match="invalid run id"):
        run_benchmark(
            repo_root=tmp_path,
            suite="smoke",
            agents=[fake_agent("pass")],
            tasks=tasks,
            runs_dir=tmp_path / "runs",
            timeout_s=5,
            run_id="../outside",
        )

    assert not (tmp_path / "outside").exists()


def test_run_benchmark_uses_live_executor_by_default(tmp_path: Path) -> None:
    from heist.runner import LiveExecutor

    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    # The Protocol-typed param must be optional and default to LiveExecutor.
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="default-executor-run",
    )
    assert manifest.kind == "live"
    assert manifest.source_run_id is None
    assert isinstance(LiveExecutor(), object)  # smoke: class is importable
    assert results[0].success is True


def test_run_benchmark_accepts_explicit_live_executor(tmp_path: Path) -> None:
    from heist.runner import LiveExecutor

    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="explicit-executor-run",
        executor=LiveExecutor(),
    )
    # Same outcome as the default-executor path.
    assert manifest.kind == "live"
    assert results[0].success is True


def test_make_run_id_is_strictly_monotonic_in_tight_loop() -> None:
    # Real invariant: even back-to-back calls within the same microsecond must
    # produce distinct, lexically-sortable ids. The 4-hex random suffix carries
    # the burden when the timestamp part repeats.
    ids = [make_run_id() for _ in range(50)]
    assert len(set(ids)) == 50, "make_run_id collided in a tight loop"
    # Without the random suffix, tight loops can produce equal ids and the
    # `<= sorted` check would pass trivially. Assert strict order.
    from itertools import pairwise

    for prev, curr in pairwise(ids):
        assert prev < curr or prev[:22] == curr[:22], (prev, curr)


def test_cheating_run_is_invalidated(tmp_path: Path) -> None:
    # An agent whose transcript references the task's hidden grader path is
    # treated as contaminated: the grader is skipped and the row is errored.
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    hidden = tmp_path / "tasks" / "smoke" / "marker" / "hidden"
    cheater = AgentSpec(
        id="fake-cheater",
        label="Cheater",
        provider="fake",
        model_id="fake-model",
        command=[sys.executable, str(FAKE_AGENT_SCRIPT), "cheat"],
        env_overrides={"HEIST_CHEAT_ECHO": str(hidden)},
    )
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[cheater],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="cheat-run",
        jobs=1,
    )
    result = results[0]
    assert result.cheating_detected is True
    assert result.outcome_status == "errored"
    assert result.success is None
    assert "cheating-detected" in (result.error or "")


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_blocked_attempt_under_sandbox_is_graded_and_flagged(tmp_path: Path) -> None:
    # Under --sandbox the agent names the answer-key path but the read is denied,
    # so the run is still graded — the score is honest — and the thwarted attempt
    # is recorded rather than invalidating the row.
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    hidden = tmp_path / "tasks" / "smoke" / "marker" / "hidden"
    blocked = AgentSpec(
        id="fake-blocked",
        label="Blocked",
        provider="fake",
        model_id="fake-model",
        command=[sys.executable, str(FAKE_AGENT_SCRIPT), "cheat"],
        env_overrides={"HEIST_CHEAT_ECHO": f"{hidden}/grader.py"},
    )
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[blocked],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=15,
        run_id="blocked-run",
        jobs=1,
        sandbox=True,
    )
    result = results[0]
    assert result.cheating_detected is False
    assert result.attempted_grader_read is True
    assert result.outcome_status == "graded"
    assert result.success is True


def test_honest_run_is_not_flagged(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="honest-run",
        jobs=1,
    )
    assert results[0].cheating_detected is False
    assert results[0].success is True


def test_execute_agent_isolates_opencode_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The selected agent declares required_env=['OPENROUTER_API_KEY']; without it
    # execute_agent raises MissingAgentEnv before reaching the patched call.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    write_marker_task(tmp_path)
    task = load_tasks("smoke", repo_root=tmp_path)[0]
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifact_dir = tmp_path / "art"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, env: dict[str, str], **kwargs: object) -> SubprocessResult:
        captured["env"] = env
        captured["command"] = command
        return SubprocessResult(stdout=b"", stderr=b"", returncode=0, timed_out=False)

    monkeypatch.setattr("heist.runner.run_subprocess_safely", fake_run)
    agent = DEFAULT_AGENTS["openrouter-deepseek-v4-pro"]
    execute_agent(agent, task, workspace, artifact_dir, timeout_s=5)
    agent_home = artifact_dir / ".agent_home"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["XDG_DATA_HOME"] == str(agent_home / "xdg-data")
    assert env["XDG_CACHE_HOME"] == str(agent_home / "xdg-cache")
    assert env["XDG_STATE_HOME"] == str(agent_home / "xdg-state")
    assert env["HEIST_WORKSPACE"] == str(workspace.resolve())
    command = captured["command"]
    assert isinstance(command, list)
    dir_index = command.index("--dir")
    assert command[dir_index + 1] == str(workspace.resolve())


def test_run_benchmark_removes_agent_home_but_keeps_artifacts(tmp_path: Path) -> None:
    # The opencode/openrouter agents wire XDG dirs into .agent_home, which the
    # CLI fills with uv archives, snapshots, and logs (hundreds of MB per pair).
    # After the run the harness must delete .agent_home while preserving every
    # real artifact (stdout/stderr/diff/grader/results).
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    agent = AgentSpec(
        id="fake-writes_cache",
        label="Fake writes_cache",
        provider="fake",
        model_id="fake-model",
        command=[sys.executable, str(FAKE_AGENT_SCRIPT), "writes_cache"],
        env_overrides={"XDG_CACHE_HOME": "{agent_home}/xdg-cache"},
    )
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[agent],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="cleanup-run",
        jobs=1,
    )

    assert results[0].success is True
    run_dir = Path(manifest.run_dir)
    artifact_dir = run_dir / "artifacts" / "fake-writes_cache" / tasks[0].id
    assert not (artifact_dir / ".agent_home").exists()
    assert (artifact_dir / "stdout.txt").exists()
    assert (artifact_dir / "stderr.txt").exists()
    assert (artifact_dir / "diff.patch").exists()
    assert (artifact_dir / "grader.json").exists()
    assert (run_dir / "results.jsonl").exists()


def test_execute_agent_removes_agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit-level guard on the cleanup itself: a file seeded under .agent_home
    # before the subprocess "runs" must be gone once execute_agent returns.
    write_marker_task(tmp_path)
    task = load_tasks("smoke", repo_root=tmp_path)[0]
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifact_dir = tmp_path / "art"

    def fake_run(command: list[str], **kwargs: object) -> SubprocessResult:
        bloat = artifact_dir / ".agent_home" / "xdg-cache" / "bloat.bin"
        bloat.parent.mkdir(parents=True, exist_ok=True)
        bloat.write_bytes(b"x" * 4096)
        return SubprocessResult(stdout=b"", stderr=b"", returncode=0, timed_out=False)

    monkeypatch.setattr("heist.runner.run_subprocess_safely", fake_run)
    execute_agent(fake_agent("pass"), task, workspace, artifact_dir, timeout_s=5)
    assert not (artifact_dir / ".agent_home").exists()


def test_execute_agent_wraps_command_in_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marker_task(tmp_path)
    task = load_tasks("smoke", repo_root=tmp_path)[0]
    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str], **kwargs: object) -> SubprocessResult:
        captured["command"] = command
        return SubprocessResult(stdout=b"", stderr=b"", returncode=0, timed_out=False)

    monkeypatch.setattr("heist.runner.run_subprocess_safely", fake_run)
    agent = fake_agent("pass")
    execute_agent(agent, task, workspace, tmp_path / "art", timeout_s=5, sandbox=True)
    assert captured["command"][0] == "sandbox-exec"
    assert captured["command"][3:] == agent.command


def test_run_benchmark_sandbox_requires_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="macOS"):
        run_benchmark(
            repo_root=tmp_path,
            suite="smoke",
            agents=[fake_agent("pass")],
            tasks=tasks,
            runs_dir=tmp_path / "runs",
            timeout_s=5,
            run_id="sandbox-linux",
            jobs=1,
            sandbox=True,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_sandbox_blocks_grader_read_end_to_end(tmp_path: Path) -> None:
    # Run a real probe agent through execute_agent under sandbox-exec: it must
    # still run and write its workspace, but its attempt to read the hidden
    # grader must be denied. Without the sandbox the same read succeeds.
    write_marker_task(tmp_path)
    task = load_tasks("smoke", repo_root=tmp_path)[0]
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, pathlib\n"
        "target = pathlib.Path(os.environ['PROBE_TARGET'])\n"
        "try:\n"
        "    target.read_text()\n"
        "    pathlib.Path('answer.txt').write_text('READ_OK')\n"
        "except OSError:\n"
        "    pathlib.Path('answer.txt').write_text('READ_BLOCKED')\n"
    )
    agent = AgentSpec(
        id="probe",
        label="probe",
        provider="fake",
        model_id="fake-model",
        command=[sys.executable, str(probe)],
        env_overrides={"PROBE_TARGET": str(task.hidden_path / "grader.py")},
    )

    sandboxed = tmp_path / "ws_sandboxed"
    sandboxed.mkdir()
    execute_agent(agent, task, sandboxed, tmp_path / "art_sb", timeout_s=15, sandbox=True)
    assert (sandboxed / "answer.txt").read_text() == "READ_BLOCKED"

    direct = tmp_path / "ws_direct"
    direct.mkdir()
    execute_agent(agent, task, direct, tmp_path / "art_ns", timeout_s=15, sandbox=False)
    assert (direct / "answer.txt").read_text() == "READ_OK"


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_run_benchmark_under_sandbox_completes_and_grades(tmp_path: Path) -> None:
    # The full CLI path with sandbox=True must still run, grade, and not
    # false-flag an honest agent: the workspace write and the (unsandboxed)
    # grader both work under the profile.
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=15,
        run_id="sandbox-ok",
        jobs=1,
        sandbox=True,
    )
    result = results[0]
    assert result.outcome_status == "graded"
    assert result.success is True
    assert result.cheating_detected is False
