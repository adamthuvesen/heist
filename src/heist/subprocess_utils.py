"""Shared subprocess plumbing for the runner and grader paths.

Lives outside `runner.py` so `tasks.py` can import it at module level
without a circular dependency on `runner.py` (which imports task helpers).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

logger = logging.getLogger("heist.subprocess")

# Bound on reaping a child after SIGKILL — the direct child should die almost
# immediately once the group is killed; past this it is wedged (uninterruptible
# sleep, or a grandchild that escaped the session holding the pipes open).
REAP_TIMEOUT_S = 5.0

# SIGTERM → SIGKILL grace for child process groups. Long enough for a
# well-behaved agent to flush logs / usage summaries; short enough that a
# stuck child doesn't delay shutdown.
KILL_GRACE_S = 0.5

# Bound on git workspace prep — init/add/commit/diff are local and should
# complete in well under a second; anything past this is a hang (corrupt
# index, NFS stall, signing prompt slipping past the scrub).
GIT_TIMEOUT_S = 30.0

# Posix-only; HEIST does not currently support Windows hosts for agent runs.
GIT_BASE_ARGS = [
    "-c",
    "commit.gpgsign=false",
    "-c",
    "gpg.program=/bin/true",
    "-c",
    "core.hooksPath=/dev/null",
]


def scrubbed_git_env() -> dict[str, str]:
    """git env that ignores user/system gitconfig and signing setup."""
    env = os.environ.copy()
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


@dataclass
class SubprocessResult:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool


def kill_process_group(pid: int) -> None:
    """Best-effort: SIGTERM the group, brief grace, then SIGKILL survivors.
    PermissionError after SIGTERM means the group is already gone (zombie /
    reaped) — also a successful outcome."""
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(KILL_GRACE_S)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def tail_bytes(path: Path, max_bytes: int) -> bytes:
    """Read at most the trailing `max_bytes` of `path`. Used to bound RAM when
    re-reading streamed agent output for usage parsing."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return b""
    if size <= max_bytes:
        return path.read_bytes()
    with path.open("rb") as handle:
        handle.seek(size - max_bytes)
        return handle.read()


def _register_process_group(
    process: subprocess.Popen[bytes],
    pgid_registry: set[int] | None,
    pgid_lock: threading.Lock | None,
) -> int | None:
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return None
    if pgid_registry is not None and pgid_lock is not None:
        with pgid_lock:
            pgid_registry.add(pgid)
    return pgid


def _discard_process_group(
    pgid: int | None,
    pgid_registry: set[int] | None,
    pgid_lock: threading.Lock | None,
) -> None:
    if pgid is not None and pgid_registry is not None and pgid_lock is not None:
        with pgid_lock:
            pgid_registry.discard(pgid)


def _reap_after_timeout(process: subprocess.Popen[bytes], pgid: int | None) -> tuple[bytes, bytes]:
    try:
        return process.communicate(timeout=REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # SIGKILL hasn't taken effect within the grace window — either a
        # grandchild escaped the session and is holding the pipes open, or the
        # child is wedged. Re-issue SIGKILL to the group and reap the direct
        # child so it isn't left as a zombie.
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
        try:
            process.wait(timeout=REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "could not reap child pid=%s (pgid=%s) after SIGKILL; leaving it for the OS",
                process.pid,
                pgid,
            )
        return b"", b""


def _communicate_with_timeout(
    *,
    process: subprocess.Popen[bytes],
    input_text: str | None,
    timeout_s: float,
    pgid: int | None,
) -> tuple[bytes, bytes, bool]:
    try:
        out_bytes, err_bytes = process.communicate(
            input=input_text.encode() if input_text is not None else None,
            timeout=timeout_s,
        )
        return out_bytes, err_bytes, False
    except subprocess.TimeoutExpired:
        kill_process_group(process.pid)
        out_bytes, err_bytes = _reap_after_timeout(process, pgid)
        return out_bytes, err_bytes, True


def _output_target(files: contextlib.ExitStack, path: Path | None) -> IO[bytes] | int:
    if path is None:
        return subprocess.PIPE
    path.parent.mkdir(parents=True, exist_ok=True)
    return files.enter_context(path.open("wb"))


def _start_process(
    command: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    input_text: str | None,
    stdout_target: IO[bytes] | int,
    stderr_target: IO[bytes] | int,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=stdout_target,
        stderr=stderr_target,
        start_new_session=True,
    )


def run_subprocess_safely(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout_s: float,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    pgid_registry: set[int] | None = None,
    pgid_lock: threading.Lock | None = None,
    tail_bytes_cap: int = 5_000_000,
) -> SubprocessResult:
    """Run a subprocess with a real, group-aware timeout.

    Posix-only: relies on `start_new_session=True` so the entire process tree
    can be killed via `killpg` on TimeoutExpired (subprocess.run only kills the
    direct child). Optionally streams stdout/stderr straight to disk to avoid
    buffering long-running agent output in RAM, and exposes the live process
    group id via `pgid_registry` so an external abort path (e.g. --fail-fast)
    can cancel in-flight work.
    """
    with contextlib.ExitStack() as files:
        # If Popen raises (missing binary, permission denied, fork failure)
        # the ExitStack still closes the stdout/stderr files we just opened.
        process = _start_process(
            command,
            cwd=cwd,
            env=env,
            input_text=input_text,
            stdout_target=_output_target(files, stdout_path),
            stderr_target=_output_target(files, stderr_path),
        )

        pgid = _register_process_group(process, pgid_registry, pgid_lock)

        try:
            out_bytes, err_bytes, timed_out = _communicate_with_timeout(
                process=process,
                input_text=input_text,
                timeout_s=timeout_s,
                pgid=pgid,
            )
        finally:
            _discard_process_group(pgid, pgid_registry, pgid_lock)

    if stdout_path is not None:
        out_bytes = tail_bytes(stdout_path, tail_bytes_cap)
    if stderr_path is not None:
        err_bytes = tail_bytes(stderr_path, tail_bytes_cap)
    out_bytes = out_bytes or b""
    err_bytes = err_bytes or b""

    return SubprocessResult(
        stdout=out_bytes,
        stderr=err_bytes,
        returncode=None if timed_out else process.returncode,
        timed_out=timed_out,
    )
