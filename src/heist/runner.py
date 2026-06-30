from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
import traceback
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from heist.agents import DEFAULT_AGENT_IDS
from heist.integrity import detect_grader_access, require_sandbox_supported, sandbox_wrap
from heist.models import (
    RUN_MANIFEST_SCHEMA_VERSION,
    AgentExecution,
    AgentSpec,
    CheckResult,
    GraderResult,
    RunManifest,
    RunStatus,
    TaskDefinition,
    TaskRunResult,
    UsageCapture,
)
from heist.progress import NullReporter
from heist.reporting import render_markdown
from heist.subprocess_utils import (
    GIT_BASE_ARGS,
    GIT_TIMEOUT_S,
    KILL_GRACE_S,
    run_subprocess_safely,
    scrubbed_git_env,
)
from heist.tasks import GraderInvalidOutput, copy_workspace, run_hidden_grader
from heist.usage import capture_usage, choose_cost

logger = logging.getLogger("heist.runner")

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_BenchmarkJob = tuple[int, AgentSpec, TaskDefinition]


class ReporterProtocol(Protocol):
    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None: ...
    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None: ...


class Executor(Protocol):
    """Per-(agent, task) execution seam.

    Splits "set up the workspace + cause the agent's effect" out of
    `_run_benchmark_job` so replay can substitute a captured snapshot for a
    live CLI invocation. Implementations:

    - `LiveExecutor`: copies the task workspace, baselines git, runs the
      agent CLI. Default for `heist run`.
    - `heist.replay.SnapshotExecutor`: copies a prior run's workspace,
      baselines git, constructs `AgentExecution` from the captured row
      without invoking any subprocess.

    `write_diff` is part of the protocol so a replay executor can preserve
    the source run's diff verbatim instead of producing an empty diff
    against a baseline that already contains the agent's changes.
    """

    def prepare_workspace(
        self, *, agent: AgentSpec, task: TaskDefinition, workspace: Path
    ) -> None: ...
    def run(
        self,
        *,
        agent: AgentSpec,
        task: TaskDefinition,
        workspace: Path,
        artifact_dir: Path,
        timeout_s: int,
        pgid_registry: set[int] | None,
        pgid_lock: threading.Lock | None,
    ) -> AgentExecution: ...
    def write_diff(
        self,
        *,
        agent: AgentSpec,
        task: TaskDefinition,
        workspace: Path,
        diff_path: Path,
    ) -> None: ...


class ExecutorAborted(RuntimeError):
    """Executor signal that an (agent, task) pair cannot run — produce an
    errored row with this exception's message and continue the benchmark.

    The runner short-circuits the retry loop on subclasses because retrying
    won't help: missing env vars stay missing, replay snapshots don't grow
    rows out of thin air. Subclasses identify the *category* of abort so
    downstream tooling can group failures; the runner only cares that it
    must record an errored row rather than raising.
    """


class MissingAgentEnv(ExecutorAborted):
    """Agent declared required_env that is not set in the current process."""

    def __init__(self, agent_id: str, missing: list[str]) -> None:
        super().__init__(f"agent {agent_id!r} requires env vars: {', '.join(missing)}")
        self.agent_id = agent_id
        self.missing = missing


def _require_agent_env(agent: AgentSpec) -> None:
    """Raise MissingAgentEnv if the agent's declared env vars aren't set.

    Checked before workspace prep so a misconfigured agent aborts without paying
    for a workspace copy + git baseline on every task."""
    missing = [name for name in agent.required_env if not os.environ.get(name)]
    if missing:
        raise MissingAgentEnv(agent.id, missing)


def make_run_id() -> str:
    now = datetime.now(UTC)
    suffix = secrets.token_hex(2)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond:06d}-{suffix}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run id {run_id!r}: use only letters, numbers, '.', '_', and '-'")
    return run_id


def _heist_package_root() -> Path:
    """Directory of the installed `heist` package, used to locate the harness
    git checkout regardless of the caller's CWD."""
    return Path(__file__).resolve().parent


# Bounded so a hung git (network filesystem, lock contention) doesn't block
# every benchmark run for minutes.
_HARNESS_SHA_TIMEOUT_S = 5.0


def _capture_harness_sha() -> str | None:
    """Best-effort `git rev-parse HEAD` of the heist checkout.

    Returns None when heist is installed outside a git tree, when `git` is
    missing, when the command exits non-zero, or when capture exceeds
    `_HARNESS_SHA_TIMEOUT_S`. Never raises — capture failures must not
    abort an otherwise-healthy run.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(_heist_package_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_HARNESS_SHA_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    """Atomic JSONL write: build a sibling .tmp, then os.replace into place.
    A SIGKILL or ENOSPC mid-write leaves the previous file intact instead of
    a truncated partial result that load_results would crash on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            if hasattr(row, "model_dump_json"):
                handle.write(row.model_dump_json())
            else:
                handle.write(json.dumps(row))
            handle.write("\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _baseline_workspace(workspace: Path) -> None:
    git_env = scrubbed_git_env()
    subprocess.run(
        ["git", *GIT_BASE_ARGS, "init", "-q"],
        cwd=workspace,
        check=True,
        env=git_env,
        timeout=GIT_TIMEOUT_S,
    )
    subprocess.run(
        ["git", *GIT_BASE_ARGS, "add", "."],
        cwd=workspace,
        check=True,
        env=git_env,
        timeout=GIT_TIMEOUT_S,
    )
    subprocess.run(
        [
            "git",
            *GIT_BASE_ARGS,
            "-c",
            "user.name=heist",
            "-c",
            "user.email=heist@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=workspace,
        check=True,
        env=git_env,
        timeout=GIT_TIMEOUT_S,
    )


def _write_diff(workspace: Path, diff_path: Path) -> None:
    """Capture the agent's diff as bytes. --binary handles binary file changes
    correctly; writing text would corrupt them. On failure, write a marker so
    consumers can tell 'no changes' apart from 'diff capture broke'."""
    try:
        process = subprocess.run(
            ["git", *GIT_BASE_ARGS, "diff", "--binary", "HEAD"],
            cwd=workspace,
            capture_output=True,
            check=False,
            env=scrubbed_git_env(),
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        diff_path.write_text(f"<error: git diff timed out after {GIT_TIMEOUT_S}s>\n")
        raise RuntimeError(f"git diff timed out after {GIT_TIMEOUT_S}s in {workspace}") from error
    if process.returncode != 0:
        stderr = process.stderr.decode(errors="replace").strip()
        diff_path.write_text(f"<error: git diff exit {process.returncode}>\n{stderr}\n")
        raise RuntimeError(f"git diff failed (exit {process.returncode}) in {workspace}:\n{stderr}")
    diff_path.write_bytes(process.stdout)


def _render_prompt(task: TaskDefinition, workspace: Path) -> str:
    workspace_path = str(workspace.resolve())
    return (
        f"{task.spec.prompt.strip()}\n\n"
        f"Your working directory is:\n{workspace_path}\n\n"
        "You are already inside the task workspace. Only read and edit files under that "
        "directory; do not explore parent directories or other project trees. Make the "
        "smallest code changes needed. Do not edit hidden graders. When you are done, stop; "
        "HEIST will run the grader."
    )


def _command_for_agent(agent: AgentSpec, prompt: str, workspace: Path) -> list[str]:
    workspace_path = str(workspace.resolve())
    return [
        part.replace("{prompt}", prompt).replace("{workspace}", workspace_path)
        for part in agent.command
    ]


def execute_agent(
    agent: AgentSpec,
    task: TaskDefinition,
    workspace: Path,
    artifact_dir: Path,
    timeout_s: int,
    *,
    sandbox: bool = False,
    pgid_registry: set[int] | None = None,
    pgid_lock: threading.Lock | None = None,
) -> AgentExecution:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt = _render_prompt(task, workspace)
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    command = _command_for_agent(agent, prompt, workspace)
    if sandbox:
        # Deny the agent reads of the whole tasks/ tree (hidden graders +
        # references) so it cannot read the answer key off disk.
        command = sandbox_wrap(command, task.hidden_path.parents[2])

    _require_agent_env(agent)

    agent_home = artifact_dir / ".agent_home"
    agent_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HEIST_TASK_ID"] = task.id
    env["HEIST_AGENT_ID"] = agent.id
    env["HEIST_WORKSPACE"] = str(workspace.resolve())
    for key, template in agent.env_overrides.items():
        # str.replace, not str.format: a `.format` call raises KeyError on any
        # unrelated `{` in the template (e.g. an inline JSON env value).
        env[key] = template.replace("{agent_home}", str(agent_home))

    started = time.monotonic()
    result = run_subprocess_safely(
        command,
        cwd=workspace,
        env=env,
        input_text=prompt if agent.prompt_via_stdin else None,
        timeout_s=timeout_s,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        pgid_registry=pgid_registry,
        pgid_lock=pgid_lock,
    )
    latency_s = time.monotonic() - started

    if result.timed_out:
        with stderr_path.open("ab") as handle:
            handle.write(f"\nHEIST timed out after {timeout_s}s.\n".encode())

    stdout_text = result.stdout.decode(errors="replace")
    stderr_text = result.stderr.decode(errors="replace")
    capture = capture_usage(f"{stdout_text}\n{stderr_text}")

    # The agent's XDG dirs under .agent_home accumulate uv package archives,
    # opencode snapshots, and logs — hundreds of MB per (agent, task) pair.
    # stdout/stderr (the real debug surface) are already captured to disk and
    # the run's outputs live in artifact_dir, so .agent_home is disposable.
    shutil.rmtree(agent_home, ignore_errors=True)

    return AgentExecution(
        exit_code=result.returncode,
        timed_out=result.timed_out,
        latency_s=latency_s,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        usage=capture.usage,
        reported_cost_usd=capture.reported_cost_usd,
        reported_cost_source=capture.reported_cost_source,
    )


def _empty_execution(artifact_dir: Path) -> AgentExecution:
    """An AgentExecution stand-in for failures that happened before the agent ran."""
    return AgentExecution(
        exit_code=None,
        timed_out=False,
        latency_s=0.0,
        stdout_path=str(artifact_dir / "stdout.txt"),
        stderr_path=str(artifact_dir / "stderr.txt"),
    )


def _cost_fields(agent: AgentSpec, execution: AgentExecution) -> dict[str, object]:
    capture = UsageCapture(
        usage=execution.usage,
        reported_cost_usd=execution.reported_cost_usd,
        reported_cost_source=execution.reported_cost_source,
    )
    cost, source, reconstructed, provenance = choose_cost(agent.model_id, capture)
    return {
        "reconstructed_per_task_cost_usd": reconstructed,
        "reported_session_cost_usd": capture.reported_cost_usd,
        "cost_provenance": provenance,
        "cost_usd": cost,
        "cost_source": source,
    }


@dataclass
class _Outcome:
    """Six fields that differ between `_errored_result` and `_graded_result` —
    everything else in TaskRunResult is identical between the two paths."""

    success: bool | None
    partial_credit: float | None
    outcome_status: str
    score: float
    checks: list[CheckResult]
    error: str | None = None


def _build_task_run_result(
    *,
    manifest: RunManifest,
    agent: AgentSpec,
    task: TaskDefinition,
    workspace: Path,
    diff_path: Path,
    grader_path: Path,
    execution: AgentExecution,
    outcome: _Outcome,
    cheating_detected: bool = False,
    attempted_grader_read: bool = False,
) -> TaskRunResult:
    return TaskRunResult(
        run_id=manifest.run_id,
        agent_id=agent.id,
        agent_label=agent.label,
        model_id=agent.model_id,
        suite=manifest.suite,
        task_id=task.id,
        task_title=task.spec.title,
        task_category=task.spec.category,
        success=outcome.success,
        partial_credit=outcome.partial_credit,
        outcome_status=outcome.outcome_status,  # type: ignore[arg-type]
        score=outcome.score,
        checks=outcome.checks,
        latency_s=execution.latency_s,
        tokens_in=execution.usage.input,
        tokens_out=execution.usage.output,
        tokens_in_by_model={agent.model_id: execution.usage.input},
        tokens_out_by_model={agent.model_id: execution.usage.output},
        **_cost_fields(agent, execution),
        agent_exit_code=execution.exit_code,
        timed_out=execution.timed_out,
        workspace_path=str(workspace),
        diff_path=str(diff_path),
        grader_path=str(grader_path),
        stdout_path=execution.stdout_path,
        stderr_path=execution.stderr_path,
        error=outcome.error,
        cheating_detected=cheating_detected,
        attempted_grader_read=attempted_grader_read,
    )


def _errored_result(
    *,
    manifest: RunManifest,
    agent: AgentSpec,
    task: TaskDefinition,
    workspace: Path,
    diff_path: Path,
    grader_path: Path,
    execution: AgentExecution,
    error: str,
    cheating_detected: bool = False,
    attempted_grader_read: bool = False,
) -> TaskRunResult:
    return _build_task_run_result(
        manifest=manifest,
        agent=agent,
        task=task,
        workspace=workspace,
        diff_path=diff_path,
        grader_path=grader_path,
        execution=execution,
        outcome=_Outcome(
            success=None,
            partial_credit=None,
            outcome_status="errored",
            score=0.0,
            checks=[CheckResult(name="agent", passed=False, message=error)],
            error=error,
        ),
        cheating_detected=cheating_detected,
        attempted_grader_read=attempted_grader_read,
    )


def _graded_result(
    *,
    manifest: RunManifest,
    agent: AgentSpec,
    task: TaskDefinition,
    workspace: Path,
    diff_path: Path,
    grader_path: Path,
    execution: AgentExecution,
    grader: GraderResult,
    attempted_grader_read: bool = False,
) -> TaskRunResult:
    return _build_task_run_result(
        manifest=manifest,
        agent=agent,
        task=task,
        workspace=workspace,
        diff_path=diff_path,
        grader_path=grader_path,
        execution=execution,
        outcome=_Outcome(
            success=grader.score >= 0.999,
            partial_credit=grader.score,
            outcome_status="graded",
            score=grader.score,
            checks=grader.checks,
        ),
        attempted_grader_read=attempted_grader_read,
    )


@dataclass(frozen=True)
class _JobContext:
    manifest: RunManifest
    agent: AgentSpec
    task: TaskDefinition
    workspace: Path
    artifact_dir: Path
    grader_path: Path
    diff_path: Path

    @classmethod
    def create(
        cls, *, manifest: RunManifest, agent: AgentSpec, task: TaskDefinition, run_dir: Path
    ) -> _JobContext:
        safe_agent = agent.id.replace("/", "_")
        workspace = run_dir / "workspaces" / safe_agent / task.id
        artifact_dir = run_dir / "artifacts" / safe_agent / task.id
        return cls(
            manifest=manifest,
            agent=agent,
            task=task,
            workspace=workspace,
            artifact_dir=artifact_dir,
            grader_path=artifact_dir / "grader.json",
            diff_path=artifact_dir / "diff.patch",
        )

    def empty_execution(self) -> AgentExecution:
        return _empty_execution(self.artifact_dir)

    def errored(
        self,
        *,
        execution: AgentExecution,
        error: str,
        cheating_detected: bool = False,
        attempted_grader_read: bool = False,
    ) -> TaskRunResult:
        return _errored_result(
            manifest=self.manifest,
            agent=self.agent,
            task=self.task,
            workspace=self.workspace,
            diff_path=self.diff_path,
            grader_path=self.grader_path,
            execution=execution,
            error=error,
            cheating_detected=cheating_detected,
            attempted_grader_read=attempted_grader_read,
        )

    def graded(
        self,
        *,
        execution: AgentExecution,
        grader: GraderResult,
        attempted_grader_read: bool = False,
    ) -> TaskRunResult:
        return _graded_result(
            manifest=self.manifest,
            agent=self.agent,
            task=self.task,
            workspace=self.workspace,
            diff_path=self.diff_path,
            grader_path=self.grader_path,
            execution=execution,
            grader=grader,
            attempted_grader_read=attempted_grader_read,
        )


class LiveExecutor:
    """Default `Executor` — copies the task workspace and invokes the agent CLI.

    Equivalent to the inline `copy_workspace + _baseline_workspace +
    execute_agent + _write_diff` flow used before the executor seam was
    introduced. A single instance is reused across all jobs; `sandbox` (macOS
    only) wraps the agent CLI in a `sandbox-exec` profile that denies reads of
    the repo `tasks/` tree, so the agent cannot read the hidden grader or
    reference solution off disk.
    """

    def __init__(self, *, sandbox: bool = False) -> None:
        self._sandbox = sandbox

    def prepare_workspace(self, *, agent: AgentSpec, task: TaskDefinition, workspace: Path) -> None:
        # Validate required env before the workspace copy + git baseline so a
        # misconfigured agent aborts without paying that cost on every task.
        # (Replay's executor reproduces missing-env from the snapshot instead.)
        _require_agent_env(agent)
        copy_workspace(task, workspace)
        _baseline_workspace(workspace)

    def run(
        self,
        *,
        agent: AgentSpec,
        task: TaskDefinition,
        workspace: Path,
        artifact_dir: Path,
        timeout_s: int,
        pgid_registry: set[int] | None,
        pgid_lock: threading.Lock | None,
    ) -> AgentExecution:
        return execute_agent(
            agent=agent,
            task=task,
            workspace=workspace,
            artifact_dir=artifact_dir,
            timeout_s=timeout_s,
            sandbox=self._sandbox,
            pgid_registry=pgid_registry,
            pgid_lock=pgid_lock,
        )

    def write_diff(
        self,
        *,
        agent: AgentSpec,
        task: TaskDefinition,
        workspace: Path,
        diff_path: Path,
    ) -> None:
        del agent, task
        _write_diff(workspace, diff_path)


def _run_benchmark_job(
    *,
    manifest: RunManifest,
    agent: AgentSpec,
    task: TaskDefinition,
    run_dir: Path,
    timeout_s: int,
    executor: Executor,
    retry: int = 0,
    pgid_registry: set[int] | None = None,
    pgid_lock: threading.Lock | None = None,
    sandbox: bool = False,
) -> TaskRunResult:
    ctx = _JobContext.create(manifest=manifest, agent=agent, task=task, run_dir=run_dir)
    effective_timeout_s = task.spec.timeout_s or timeout_s
    try:
        execution = _execute_with_retries(
            ctx=ctx,
            executor=executor,
            timeout_s=effective_timeout_s,
            retry=retry,
            pgid_registry=pgid_registry,
            pgid_lock=pgid_lock,
        )
    except ExecutorAborted as error:
        return _executor_aborted_result(ctx, error)

    diff_error = _diff_error_result(ctx=ctx, executor=executor, execution=execution)
    if diff_error is not None:
        return diff_error

    integrity_error, attempted_grader_read = _integrity_error_result(
        ctx=ctx, execution=execution, sandbox=sandbox
    )
    if integrity_error is not None:
        return integrity_error

    agent_error = _agent_error_result(
        ctx=ctx,
        execution=execution,
        effective_timeout_s=effective_timeout_s,
        attempted_grader_read=attempted_grader_read,
    )
    if agent_error is not None:
        return agent_error

    return _grade_job(
        ctx=ctx,
        execution=execution,
        pgid_registry=pgid_registry,
        pgid_lock=pgid_lock,
        attempted_grader_read=attempted_grader_read,
    )


def _execute_with_retries(
    *,
    ctx: _JobContext,
    executor: Executor,
    timeout_s: int,
    retry: int,
    pgid_registry: set[int] | None,
    pgid_lock: threading.Lock | None,
) -> AgentExecution:
    attempts = max(1, retry + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            executor.prepare_workspace(agent=ctx.agent, task=ctx.task, workspace=ctx.workspace)
            return executor.run(
                agent=ctx.agent,
                task=ctx.task,
                workspace=ctx.workspace,
                artifact_dir=ctx.artifact_dir,
                timeout_s=timeout_s,
                pgid_registry=pgid_registry,
                pgid_lock=pgid_lock,
            )
        except ExecutorAborted:
            raise
        except Exception as error:
            last_exc = error
            logger.warning(
                "agent invocation failed (attempt %d/%d) for %s on %s: %s",
                attempt,
                attempts,
                ctx.agent.id,
                ctx.task.id,
                error,
            )
            if attempt >= attempts:
                raise
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError(f"unreachable: {last_exc}")


def _executor_aborted_result(ctx: _JobContext, error: ExecutorAborted) -> TaskRunResult:
    # Don't retry — these are terminal-for-this-pair signals (missing env,
    # replay source missing). Record an errored row and let the run continue.
    logger.error("%s", error)
    ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
    ctx.diff_path.write_text("")
    return ctx.errored(execution=ctx.empty_execution(), error=str(error))


def _diff_error_result(
    *, ctx: _JobContext, executor: Executor, execution: AgentExecution
) -> TaskRunResult | None:
    try:
        executor.write_diff(
            agent=ctx.agent,
            task=ctx.task,
            workspace=ctx.workspace,
            diff_path=ctx.diff_path,
        )
    except RuntimeError as error:
        logger.error("git diff capture failed for %s on %s: %s", ctx.agent.id, ctx.task.id, error)
        return ctx.errored(
            execution=execution,
            error=str(error),
        )
    return None


def _integrity_error_result(
    *, ctx: _JobContext, execution: AgentExecution, sandbox: bool
) -> tuple[TaskRunResult | None, bool]:
    access = detect_grader_access(
        execution.stdout_path, execution.stderr_path, ctx.task, sandboxed=sandbox
    )
    if access.contaminated is not None:
        logger.error("integrity: %s on %s — %s", ctx.agent.id, ctx.task.id, access.contaminated)
        return ctx.errored(
            execution=execution,
            error=f"cheating-detected: {access.contaminated}",
            cheating_detected=True,
            attempted_grader_read=True,
        ), True
    attempted = access.attempted is not None
    if attempted:
        logger.warning("integrity: %s on %s — %s", ctx.agent.id, ctx.task.id, access.attempted)
    return None, attempted


def _agent_error_result(
    *,
    ctx: _JobContext,
    execution: AgentExecution,
    effective_timeout_s: int,
    attempted_grader_read: bool,
) -> TaskRunResult | None:
    if execution.timed_out:
        return ctx.errored(
            execution=execution,
            error=f"agent timed out after {effective_timeout_s}s",
            attempted_grader_read=attempted_grader_read,
        )

    if execution.exit_code not in (0, None):
        # A negative code is a signal death (e.g. -15 SIGTERM from a --fail-fast
        # abort, or a crash); report it as such rather than as a bogus exit code
        # so harness-initiated kills aren't mistaken for the agent's own failure.
        code = execution.exit_code
        detail = f"killed by signal {-code}" if code < 0 else f"exited with code {code}"
        return ctx.errored(
            execution=execution,
            error=f"agent {detail}",
            attempted_grader_read=attempted_grader_read,
        )
    return None


def _grade_job(
    *,
    ctx: _JobContext,
    execution: AgentExecution,
    pgid_registry: set[int] | None,
    pgid_lock: threading.Lock | None,
    attempted_grader_read: bool,
) -> TaskRunResult:
    try:
        grader = run_hidden_grader(
            ctx.task,
            ctx.workspace,
            pgid_registry=pgid_registry,
            pgid_lock=pgid_lock,
        )
        ctx.grader_path.write_text(grader.model_dump_json(indent=2))
        return ctx.graded(
            execution=execution,
            grader=grader,
            attempted_grader_read=attempted_grader_read,
        )
    except Exception as error:
        kind = (
            "grader_invalid_output" if isinstance(error, GraderInvalidOutput) else "grader_failed"
        )
        first_line = str(error).splitlines()[0] if str(error) else error.__class__.__name__
        ctx.grader_path.write_text(
            json.dumps(
                {
                    "error_kind": kind,
                    "error": first_line,
                    "traceback": "".join(traceback.format_exception(error)),
                },
                indent=2,
            )
        )
        return ctx.errored(
            execution=execution,
            error=first_line,
            attempted_grader_read=attempted_grader_read,
        )


def _ordered_completed_results(
    results_by_index: dict[int, TaskRunResult], total_jobs: int
) -> list[TaskRunResult]:
    return [results_by_index[index] for index in range(total_jobs) if index in results_by_index]


def _ordered_jobs(agents: list[AgentSpec], tasks: list[TaskDefinition]) -> list[_BenchmarkJob]:
    jobs: list[_BenchmarkJob] = []
    for agent in agents:
        for task in tasks:
            jobs.append((len(jobs), agent, task))
    return jobs


def _resolve_provider_caps(
    agents: list[AgentSpec],
    *,
    jobs: int,
    provider_jobs: dict[str, int] | None,
) -> dict[str, int]:
    """Return the per-provider concurrency cap for each provider in `agents`.

    Default: every provider gets the global `jobs` cap. Explicit per-provider
    caps from `provider_jobs` override that. Caps are clipped to `jobs` (the
    global cap is the hard upper bound).
    """
    providers = {agent.provider for agent in agents}
    caps: dict[str, int] = {provider: jobs for provider in providers}
    for provider, cap in (provider_jobs or {}).items():
        if cap < 1:
            raise ValueError(f"provider_jobs[{provider!r}] must be >= 1, got {cap}")
        if provider in caps:
            caps[provider] = min(cap, jobs)
    return caps


@dataclass
class _AbortState:
    event: threading.Event
    pgid_registry: set[int]
    pgid_lock: threading.Lock

    @classmethod
    def create(cls) -> _AbortState:
        return cls(
            event=threading.Event(),
            pgid_registry=set(),
            pgid_lock=threading.Lock(),
        )

    def is_set(self) -> bool:
        return self.event.is_set()

    def trigger(self) -> None:
        self.event.set()
        # Hold pgid_lock across the whole TERM→grace→KILL sequence so a child
        # cannot exit and have its pgid recycled by the OS between snapshot and
        # signal — the registry stays authoritative for "is this pgid still ours?".
        with self.pgid_lock:
            pgids = list(self.pgid_registry)
            for pgid in pgids:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    continue
            # Escalate to SIGKILL: a SIGTERM-ignoring agent CLI (node/python
            # wrappers under the PTY shim) would otherwise survive the abort and
            # keep burning API budget. Mirror kill_process_group's grace window.
            time.sleep(KILL_GRACE_S)
            for pgid in pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    continue


class _AbortedJob(Exception):
    """Raised inside a worker when fail-fast aborts further work."""

    def __init__(self, index: int) -> None:
        super().__init__(f"job {index} aborted before start")
        self.index = index


def _append_result(path: Path, result: TaskRunResult) -> None:
    # Append a single row in completion order. The final rewrite re-orders by
    # declared index. Previously every completion rewrote the entire file, which
    # was O(N²) bytes over a run.
    with path.open("a") as handle:
        handle.write(result.model_dump_json())
        handle.write("\n")


def _should_fail_fast_abort(result: TaskRunResult) -> bool:
    # --fail-fast covers both 'errored' (capture/grader failure) and 'graded but
    # failing'. A known-bad agent shouldn't burn the whole budget on tasks it
    # can't pass.
    return result.outcome_status == "errored" or (
        result.outcome_status == "graded" and result.success is False
    )


def _run_job_and_report(
    *,
    manifest: RunManifest,
    agent: AgentSpec,
    task: TaskDefinition,
    run_dir: Path,
    timeout_s: int,
    executor: Executor,
    retry: int,
    abort_state: _AbortState,
    reporter: ReporterProtocol,
    sandbox: bool,
) -> TaskRunResult:
    reporter.on_start(agent, task)
    result = _run_benchmark_job(
        manifest=manifest,
        agent=agent,
        task=task,
        run_dir=run_dir,
        timeout_s=timeout_s,
        executor=executor,
        retry=retry,
        pgid_registry=abort_state.pgid_registry,
        pgid_lock=abort_state.pgid_lock,
        sandbox=sandbox,
    )
    reporter.on_finish(agent, task, result)
    return result


def _run_jobs_serial(
    *,
    ordered_jobs: list[_BenchmarkJob],
    manifest: RunManifest,
    run_dir: Path,
    timeout_s: int,
    executor: Executor,
    retry: int,
    abort_state: _AbortState,
    reporter: ReporterProtocol,
    sandbox: bool,
    fail_fast: bool,
    results_path: Path,
    results_by_index: dict[int, TaskRunResult],
) -> None:
    for index, agent, task in ordered_jobs:
        if abort_state.is_set():
            break
        result = _run_job_and_report(
            manifest=manifest,
            agent=agent,
            task=task,
            run_dir=run_dir,
            timeout_s=timeout_s,
            executor=executor,
            retry=retry,
            abort_state=abort_state,
            reporter=reporter,
            sandbox=sandbox,
        )
        results_by_index[index] = result
        _append_result(results_path, result)
        if fail_fast and _should_fail_fast_abort(result):
            abort_state.trigger()


def _run_jobs_parallel(
    *,
    ordered_jobs: list[_BenchmarkJob],
    agents: list[AgentSpec],
    jobs: int,
    provider_jobs: dict[str, int] | None,
    manifest: RunManifest,
    run_dir: Path,
    timeout_s: int,
    executor: Executor,
    retry: int,
    abort_state: _AbortState,
    reporter: ReporterProtocol,
    sandbox: bool,
    fail_fast: bool,
    results_path: Path,
    results_by_index: dict[int, TaskRunResult],
) -> None:
    caps = _resolve_provider_caps(agents, jobs=jobs, provider_jobs=provider_jobs)
    global_sem = threading.BoundedSemaphore(jobs)
    results_lock = threading.Lock()
    pools: dict[str, ThreadPoolExecutor] = {
        provider: ThreadPoolExecutor(
            max_workers=caps[provider],
            thread_name_prefix=f"heist-{provider}",
        )
        for provider in caps
    }

    def _worker(index: int, agent: AgentSpec, task: TaskDefinition) -> tuple[int, TaskRunResult]:
        if abort_state.is_set():
            raise _AbortedJob(index)
        with global_sem:
            if abort_state.is_set():
                raise _AbortedJob(index)
            result = _run_job_and_report(
                manifest=manifest,
                agent=agent,
                task=task,
                run_dir=run_dir,
                timeout_s=timeout_s,
                executor=executor,
                retry=retry,
                abort_state=abort_state,
                reporter=reporter,
                sandbox=sandbox,
            )
            return index, result

    futures: dict[Future[tuple[int, TaskRunResult]], int] = {}
    try:
        for index, agent, task in ordered_jobs:
            future = pools[agent.provider].submit(_worker, index, agent, task)
            futures[future] = index

        for future in as_completed(futures):
            try:
                index, result = future.result()
            except _AbortedJob:
                continue
            except Exception:
                abort_state.trigger()
                raise
            with results_lock:
                results_by_index[index] = result
                _append_result(results_path, result)
            if fail_fast and _should_fail_fast_abort(result):
                abort_state.trigger()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)


def _finalize_run(
    *,
    manifest: RunManifest,
    run_dir: Path,
    results_path: Path,
    results_by_index: dict[int, TaskRunResult],
    total_jobs: int,
    started: float,
    abort_state: _AbortState,
) -> tuple[RunManifest, list[TaskRunResult]]:
    # Rewrite results.jsonl in declared-index order so consumers that read the
    # file at end-of-run see a deterministic order regardless of completion
    # order. The atomic write also closes any partial append from a crashed
    # final row.
    results = _ordered_completed_results(results_by_index, total_jobs)
    write_jsonl(results_path, results)
    # Finalize the manifest no matter how we exited so consumers can distinguish
    # a completed run from a crash/abort.
    status: RunStatus = "aborted" if abort_state.is_set() else "completed"
    updated_manifest = manifest.model_copy(
        update={
            "completed_at": datetime.now(UTC),
            "duration_s": time.monotonic() - started,
            "status": status,
        }
    )
    _write_manifest(run_dir, updated_manifest)
    return updated_manifest, results


def _create_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Run already exists: {run_dir}") from error
    return run_dir


def _initial_manifest(
    *,
    repo_root: Path,
    suite: str,
    agents: list[AgentSpec],
    tasks: list[TaskDefinition],
    run_id: str,
    run_dir: Path,
    kind: str,
    source_run_id: str | None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        suite=suite,
        agent_ids=[agent.id for agent in agents],
        task_ids=[task.id for task in tasks],
        repo_root=str(repo_root),
        run_dir=str(run_dir),
        default_agents=DEFAULT_AGENT_IDS,
        harness_git_sha=_capture_harness_sha(),
        kind=kind,  # type: ignore[arg-type]
        source_run_id=source_run_id,
    )


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _prepare_results_path(run_dir: Path) -> Path:
    results_path = run_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Touch the file so consumers tailing it during the run don't trip on
    # "file not found" before the first job completes.
    results_path.touch()
    return results_path


def run_benchmark(
    *,
    repo_root: Path,
    suite: str,
    agents: list[AgentSpec],
    tasks: list[TaskDefinition],
    runs_dir: Path,
    timeout_s: int,
    run_id: str | None = None,
    jobs: int = 1,
    provider_jobs: dict[str, int] | None = None,
    reporter: ReporterProtocol | None = None,
    retry: int = 0,
    fail_fast: bool = False,
    executor: Executor | None = None,
    sandbox: bool = False,
    kind: str = "live",
    source_run_id: str | None = None,
) -> tuple[RunManifest, list[TaskRunResult]]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if retry < 0:
        raise ValueError("retry must be >= 0")
    if sandbox:
        require_sandbox_supported()

    reporter = reporter or NullReporter()
    executor = executor or LiveExecutor(sandbox=sandbox)

    run_id = validate_run_id(run_id or make_run_id())
    run_dir = _create_run_dir(runs_dir, run_id)
    started = time.monotonic()
    manifest = _initial_manifest(
        repo_root=repo_root,
        suite=suite,
        agents=agents,
        tasks=tasks,
        run_id=run_id,
        run_dir=run_dir,
        kind=kind,
        source_run_id=source_run_id,
    )
    _write_manifest(run_dir, manifest)

    ordered_jobs = _ordered_jobs(agents, tasks)
    results_by_index: dict[int, TaskRunResult] = {}
    abort_state = _AbortState.create()
    results_path = _prepare_results_path(run_dir)

    try:
        if jobs == 1:
            _run_jobs_serial(
                ordered_jobs=ordered_jobs,
                manifest=manifest,
                run_dir=run_dir,
                timeout_s=timeout_s,
                executor=executor,
                retry=retry,
                abort_state=abort_state,
                reporter=reporter,
                sandbox=sandbox,
                fail_fast=fail_fast,
                results_path=results_path,
                results_by_index=results_by_index,
            )
        else:
            _run_jobs_parallel(
                ordered_jobs=ordered_jobs,
                agents=agents,
                jobs=jobs,
                provider_jobs=provider_jobs,
                manifest=manifest,
                run_dir=run_dir,
                timeout_s=timeout_s,
                executor=executor,
                retry=retry,
                abort_state=abort_state,
                reporter=reporter,
                sandbox=sandbox,
                fail_fast=fail_fast,
                results_path=results_path,
                results_by_index=results_by_index,
            )
    finally:
        manifest, results = _finalize_run(
            manifest=manifest,
            run_dir=run_dir,
            results_path=results_path,
            results_by_index=results_by_index,
            total_jobs=len(ordered_jobs),
            started=started,
            abort_state=abort_state,
        )

    return manifest, results


def load_results(run_dir: Path) -> list[TaskRunResult]:
    return [TaskRunResult.model_validate(row) for row in read_jsonl(run_dir / "results.jsonl")]


_V1_TO_V2_DEFAULTS: dict[str, object] = {
    "harness_git_sha": None,
    "tags": [],
    "source_run_id": None,
    "kind": "live",
}


def _migrate_manifest_payload(raw: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Upgrade manifest payloads to RUN_MANIFEST_SCHEMA_VERSION.

    Returns (payload, mutated). When `mutated` is True the caller should
    persist the new payload back to disk so the next load is a no-op.
    """
    version = raw.get("schema_version")
    if version == RUN_MANIFEST_SCHEMA_VERSION:
        return raw, False
    if version == 1:
        upgraded = dict(raw)
        upgraded["schema_version"] = 2
        for key, default in _V1_TO_V2_DEFAULTS.items():
            upgraded.setdefault(key, default)
        return upgraded, True
    raise ValueError(
        f"manifest schema_version={version!r} is incompatible "
        f"with this version of heist (expects {RUN_MANIFEST_SCHEMA_VERSION}). "
        f"Re-run the benchmark or regenerate the manifest."
    )


def load_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / "manifest.json"
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"manifest at {path} is not a JSON object")
    payload, mutated = _migrate_manifest_payload(raw)
    if mutated:
        path.write_text(json.dumps(payload, indent=2))
    return RunManifest.model_validate(payload)


def regrade_run(run_dir: Path, tasks: list[TaskDefinition]) -> list[TaskRunResult]:
    manifest = load_manifest(run_dir)
    task_by_id = {task.id: task for task in tasks}
    results = load_results(run_dir)
    regraded: list[TaskRunResult] = []
    for result in results:
        task = task_by_id.get(result.task_id)
        if task is None:
            raise ValueError(
                f"run {manifest.run_id!r} references task {result.task_id!r}, "
                f"but suite {manifest.suite!r} does not contain that task"
            )
        workspace = Path(result.workspace_path)
        if not workspace.exists():
            # The agent never produced a workspace (e.g. missing-env or an abort
            # before setup). There's nothing to grade — carry the original row
            # forward unchanged rather than grading an empty tree into a bogus
            # 0.0 'graded' row that would mask the real failure.
            regraded.append(result)
            continue
        grader = run_hidden_grader(task, workspace)
        updated = result.model_copy(
            update={
                "success": grader.score >= 0.999,
                "partial_credit": grader.score,
                "outcome_status": "graded",
                "score": grader.score,
                "checks": grader.checks,
                "error": None,
            }
        )
        regraded.append(updated)
    out_path = run_dir / "regrade-results.jsonl"
    write_jsonl(out_path, regraded)
    (run_dir / "regrade-manifest.json").write_text(manifest.model_dump_json(indent=2))
    # Re-render a summary alongside the regraded rows so report consumers can
    # diff it against the original summary.md.
    (run_dir / "regrade-summary.md").write_text(render_markdown(regraded))
    return regraded
