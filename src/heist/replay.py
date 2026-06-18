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

    def _terminal_missing_env(self, agent_id: str, task_id: str) -> MissingAgentEnv | None:
        """Reconstruct a MissingAgentEnv abort from a source row, if it recorded
        one. Such rows have no workspace (the live run now aborts before the
        workspace copy), so this must be detected before the source-workspace
        check — otherwise replay reports 'source workspace missing' instead of
        reproducing the original env error."""
        row = self._rows_by_pair.get((agent_id, task_id))
        if row is None or row.outcome_status != "errored" or not row.error:
            return None
        match = _MISSING_ENV_RE.match(row.error)
        if match is None:
            return None
        missing = [v.strip() for v in match.group("vars").split(",")]
        return MissingAgentEnv(match.group("agent_id"), missing)

    def prepare_workspace(self, *, agent: AgentSpec, task: TaskDefinition, workspace: Path) -> None:
        abort = self._terminal_missing_env(agent.id, task.id)
        if abort is not None:
            raise abort
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

        # Faithful terminal-outcome preservation: missing-env errors re-raise so
        # `_run_benchmark_job`'s MissingAgentEnv branch produces the same outcome.
        # Normally caught earlier in prepare_workspace; kept here as a fallback.
        abort = self._terminal_missing_env(agent.id, task.id)
        if abort is not None:
            raise abort

        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_dest = artifact_dir / "stdout.txt"
        stderr_dest = artifact_dir / "stderr.txt"
        source_stdout = self._confined_source(row.stdout_path)
        source_stderr = self._confined_source(row.stderr_path)
        if source_stdout is None or not source_stdout.exists():
            raise ReplaySourceMissing(f"captured stdout missing for {pair}: {row.stdout_path}")
        if source_stderr is None or not source_stderr.exists():
            raise ReplaySourceMissing(f"captured stderr missing for {pair}: {row.stderr_path}")
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
            source_diff = self._confined_source(row.diff_path)
            if source_diff is not None and source_diff.exists():
                try:
                    diff_path.write_bytes(source_diff.read_bytes())
                    return
                except OSError:
                    pass
        diff_path.write_text("<error: replay source diff missing>\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_workspace(self, agent_id: str, task_id: str) -> Path:
        return self.source_run_dir / "workspaces" / _safe_agent(agent_id) / task_id

    def _confined_source(self, recorded: str) -> Path | None:
        """Resolve a path recorded in results.jsonl, confined to the source run
        dir. A tampered/hand-edited run file could point stdout/stderr/diff paths
        at any file on disk; an escaping path is treated as missing rather than
        copied out of the run tree."""
        root = self.source_run_dir.resolve()
        candidate = Path(recorded)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        return resolved if resolved.is_relative_to(root) else None

    # ------------------------------------------------------------------
    # Selection helpers used by the CLI
    # ------------------------------------------------------------------

    def known_pairs(self) -> set[tuple[str, str]]:
        return set(self._rows_by_pair)

    def known_agents(self) -> set[str]:
        return {pair[0] for pair in self._rows_by_pair}

    def known_tasks(self) -> set[str]:
        return {pair[1] for pair in self._rows_by_pair}
