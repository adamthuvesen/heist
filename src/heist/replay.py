"""Snapshot-based replay: re-grade and re-report a prior live run.

`SnapshotExecutor` implements the `Executor` protocol from `runner` and is
swapped in by `heist runs replay` instead of `LiveExecutor`. It does not
invoke any agent CLI — every per-task `AgentExecution` is reconstructed
from the source run's captured stdout/stderr/results.jsonl row.

Replay output is a normal `runs/<id>/` dir whose manifest records
`kind: "replay"` and `source_run_id`; the cross-run analysis surface
treats it as just another run for `list`, `compare`, and `history`.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from pathlib import Path

from heist.models import AgentExecution, AgentSpec, RunManifest, TaskDefinition, TokenUsage
from heist.runner import (
    ExecutorAborted,
    MissingAgentEnv,
    _baseline_workspace,
    load_manifest,
    load_results,
)

# Source rows surface env-missing failures with `execute_agent`'s error string,
# which embeds the message format `agent <id> requires env vars: A, B`. Matching
# on that prefix lets `SnapshotExecutor` propagate the same terminal outcome by
# re-raising `MissingAgentEnv` — no per-error type code in the runner.
_MISSING_ENV_RE = re.compile(r"agent '(?P<agent_id>[^']+)' requires env vars: (?P<vars>.+)")

logger = logging.getLogger("heist.replay")


class ReplayOfReplayError(RuntimeError):
    """Raised when the user points `replay` at a source run that is itself
    a replay. The replay's `source_run_id` is named in the message so users
    can chase back to the original live run."""


class ReplaySourceMissing(ExecutorAborted):
    """Raised when the snapshot can't materialise a workspace or execution
    for an (agent, task) pair — e.g., the source workspace dir was deleted
    or `results.jsonl` lacks the matching row. The runner converts this to
    an `_errored_result` via the existing failure-routing pattern."""


def _safe_agent(agent_id: str) -> str:
    # Mirrors `_run_benchmark_job` so replay's path layout matches live.
    return agent_id.replace("/", "_")


def _copy_source_tree(source_ws: Path, dest_ws: Path) -> None:
    """Copy source's tree into dest_ws, skipping `.git`.

    The dest gets a fresh `_baseline_workspace` initialisation afterwards, so
    `git diff HEAD` from the runner is empty for the copy itself. Replay's
    `write_diff` copies the source's diff verbatim so the artefact still
    matches what the agent produced.
    """
    if not source_ws.is_dir():
        raise ReplaySourceMissing(f"source workspace is not a directory: {source_ws}")
    dest_ws.mkdir(parents=True, exist_ok=True)
    for child in source_ws.iterdir():
        if child.name == ".git":
            continue
        target = dest_ws / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


class SnapshotExecutor:
    """`Executor` that replays a prior live run from disk.

    Construction loads the source manifest and indexes `results.jsonl` by
    (agent_id, task_id). Calls to `prepare_workspace` / `run` / `write_diff`
    materialise the per-pair snapshot lazily; nothing is mutated on the
    source side.
    """

    def __init__(self, source_run_dir: Path) -> None:
        self.source_run_dir = source_run_dir
        self.source_manifest: RunManifest = load_manifest(source_run_dir)
        if self.source_manifest.kind == "replay":
            transitive_source = self.source_manifest.source_run_id or "<unknown>"
            raise ReplayOfReplayError(
                f"refusing to replay {self.source_manifest.run_id!r}: it is itself "
                f"a replay of {transitive_source!r}. Replay against that run instead."
            )
        rows = load_results(source_run_dir)
        self._rows_by_pair = {(row.agent_id, row.task_id): row for row in rows}

    # ------------------------------------------------------------------
    # Executor protocol
    # ------------------------------------------------------------------

    def prepare_workspace(self, *, agent: AgentSpec, task: TaskDefinition, workspace: Path) -> None:
        source_ws = self._source_workspace(agent.id, task.id)
        if not source_ws.exists():
            raise ReplaySourceMissing(
                f"source workspace missing for ({agent.id}, {task.id}): {source_ws}"
            )
        # Materialise source's final state, then baseline so the grader sees
        # a git-tracked working tree. `git diff HEAD` against this baseline
        # is empty; replay's `write_diff` copies source's diff.patch
        # verbatim so the artefact still matches the agent's diff.
        _copy_source_tree(source_ws, workspace)
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
        del workspace, timeout_s, pgid_registry, pgid_lock
        pair = (agent.id, task.id)
        row = self._rows_by_pair.get(pair)
        if row is None:
            raise ReplaySourceMissing(
                f"source row missing for {pair} in {self.source_run_dir / 'results.jsonl'}"
            )

        # Faithful terminal-outcome preservation:
        # - missing-env errors must re-raise so `_run_benchmark_job`'s existing
        #   MissingAgentEnv branch produces the same error string/outcome.
        if row.outcome_status == "errored" and row.error:
            match = _MISSING_ENV_RE.match(row.error)
            if match:
                missing = [v.strip() for v in match.group("vars").split(",")]
                raise MissingAgentEnv(match.group("agent_id"), missing)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_dest = artifact_dir / "stdout.txt"
        stderr_dest = artifact_dir / "stderr.txt"
        source_stdout = Path(row.stdout_path)
        source_stderr = Path(row.stderr_path)
        if not source_stdout.exists():
            raise ReplaySourceMissing(f"captured stdout missing for {pair}: {source_stdout}")
        if not source_stderr.exists():
            raise ReplaySourceMissing(f"captured stderr missing for {pair}: {source_stderr}")
        shutil.copy2(source_stdout, stdout_dest)
        shutil.copy2(source_stderr, stderr_dest)

        # Preserve the captured AgentExecution shape. The downstream timeout
        # short-circuit in `_run_benchmark_job` reads `timed_out` here.
        return AgentExecution(
            exit_code=row.agent_exit_code,
            timed_out=row.timed_out,
            latency_s=row.latency_s or 0.0,
            stdout_path=str(stdout_dest),
            stderr_path=str(stderr_dest),
            usage=TokenUsage(
                input=row.tokens_in,
                output=row.tokens_out,
                cache_read=0,
                cache_write=0,
            ),
            reported_cost_usd=row.reported_session_cost_usd,
            reported_cost_source=("reported" if row.cost_source == "reported" else None),
        )

    def write_diff(
        self,
        *,
        agent: AgentSpec,
        task: TaskDefinition,
        workspace: Path,
        diff_path: Path,
    ) -> None:
        del workspace
        # Prefer copying the source's diff verbatim so replay artefacts match
        # the source's diff bytes. If the source's diff file is missing or
        # unreadable, fall through to writing a marker — never raise from
        # diff capture (matches LiveExecutor's failure semantics).
        row = self._rows_by_pair.get((agent.id, task.id))
        if row is not None:
            source_diff = Path(row.diff_path)
            if source_diff.exists():
                try:
                    diff_path.write_bytes(source_diff.read_bytes())
                    return
                except OSError as exc:
                    # Never raise from diff capture, but don't silently swallow
                    # the cause: an infra failure (permissions, ENOSPC) would
                    # otherwise be indistinguishable from a genuinely missing
                    # source diff once it reaches the marker below.
                    logger.warning(
                        "replay: could not copy source diff %s -> %s: %s",
                        source_diff,
                        diff_path,
                        exc,
                    )
        diff_path.write_text("<error: replay source diff missing>\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_workspace(self, agent_id: str, task_id: str) -> Path:
        return self.source_run_dir / "workspaces" / _safe_agent(agent_id) / task_id

    # ------------------------------------------------------------------
    # Selection helpers used by the CLI
    # ------------------------------------------------------------------

    def known_pairs(self) -> set[tuple[str, str]]:
        return set(self._rows_by_pair)

    def known_agents(self) -> set[str]:
        return {pair[0] for pair in self._rows_by_pair}

    def known_tasks(self) -> set[str]:
        return {pair[1] for pair in self._rows_by_pair}
