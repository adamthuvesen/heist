"""Run-integrity guards: keep an agent from reading the answer key.

The hidden grader and the reference solution live in the source tree at a fixed
path relative to the run workspace, and the agent CLI runs on the host with full
filesystem access. Two defenses live here:

- ``detect_grader_access`` — a post-run check that scans the agent transcript for
  any reference to *this task's* ``hidden/`` or ``reference/`` path. An honest
  agent never names those absolute paths, so a hit means the run is contaminated
  and must be invalidated. Always on.

- ``sandbox_wrap`` / ``require_sandbox_supported`` — wrap the agent argv in a
  macOS ``sandbox-exec`` profile that denies reads of the repo ``tasks/`` tree,
  so the answer key is simply unreadable while everything else (workspace writes,
  network for the model API and auth) keeps working. Opt-in.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heist.models import TaskDefinition

# Read at most this much from the tail of each transcript. Real agent transcripts
# are a few hundred KB; this caps memory on a pathological multi-GB log.
_TRANSCRIPT_READ_CAP = 10_000_000


@dataclass(frozen=True)
class GraderAccess:
    """Outcome of scanning a transcript for answer-key access.

    ``contaminated`` is set when the agent could have *read* the hidden grader or
    reference — the run is invalid. ``attempted`` is set when the agent named the
    path but the sandbox made the read impossible: the score is trustworthy, but
    the cheat attempt is worth recording.

    Which one fires is decided by whether the run was sandboxed, not by parsing
    the transcript: under ``sandbox-exec`` a read of ``tasks/`` is physically
    denied, so any mention is necessarily a thwarted attempt; unsandboxed, a
    mention means the agent read the file off disk.
    """

    contaminated: str | None = None
    attempted: str | None = None


def _read_tail(path: str | Path | None, cap: int = _TRANSCRIPT_READ_CAP) -> str:
    if path is None:
        return ""
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
        with file_path.open("rb") as handle:
            if size > cap:
                handle.seek(size - cap)
            data = handle.read()
    except OSError:
        return ""
    return data.decode(errors="replace")


def _path_variants(path: Path, task_id: str) -> set[str]:
    """Needles that betray a read of ``path`` for this task.

    Includes absolute and symlink-resolved paths, plus task-relative forms so a
    grader read via ``cd`` into the task dir then ``cat hidden/grader.py`` is
    still caught. This scan is best-effort; ``sandbox_wrap`` is the hard
    prevention layer.
    """
    variants = {str(path)}
    with contextlib.suppress(OSError):
        variants.add(os.path.realpath(path))
    leaf = path.name
    if leaf:
        variants.add(f"{task_id}/{leaf}")
        if leaf == "hidden":
            variants.add("hidden/grader.py")
            variants.add(f"{task_id}/hidden/grader.py")
    return {variant for variant in variants if variant}


def detect_grader_access(
    stdout_path: str | Path | None,
    stderr_path: str | Path | None,
    task: TaskDefinition,
    *,
    sandboxed: bool = False,
) -> GraderAccess:
    """Scan the agent transcript for any reference to this task's hidden grader or
    reference solution path. Under a sandbox the read could not have succeeded, so
    a hit is a thwarted ``attempted`` read; unsandboxed, a hit means the file was
    read off disk and the run is ``contaminated``."""
    transcript = f"{_read_tail(stdout_path)}\n{_read_tail(stderr_path)}"
    if not transcript.strip():
        return GraderAccess()
    for label, base in (
        ("hidden grader", task.hidden_path),
        ("reference solution", task.reference_path),
    ):
        for needle in _path_variants(Path(base), task.id):
            if needle in transcript:
                if sandboxed:
                    return GraderAccess(
                        attempted=f"agent tried to read the {label} but the sandbox "
                        f"denied it ({needle})"
                    )
                return GraderAccess(contaminated=f"agent read the {label} ({needle})")
    return GraderAccess()


def require_sandbox_supported() -> None:
    """Fail loudly rather than silently run unsandboxed on a platform that has no
    ``sandbox-exec``."""
    if sys.platform != "darwin":
        raise RuntimeError(
            "sandbox=true requires macOS (sandbox-exec). Run without --sandbox on "
            "this platform (the cheat-detector still runs), or use container "
            "isolation."
        )


def sandbox_wrap(command: list[str], tasks_root: Path) -> list[str]:
    """Wrap ``command`` in a ``sandbox-exec`` profile that denies reads of the
    repo ``tasks/`` tree (graders + references) while allowing everything else.
    Child processes inherit the sandbox."""
    if not command:
        return command
    deny = os.path.realpath(tasks_root)
    # The path is embedded in a double-quoted SBPL string; escape backslash and
    # double-quote so a path containing either can't terminate the string early
    # and silently weaken the deny rule. (defense-in-depth — repo paths rarely
    # contain these, but a security policy shouldn't be built by raw interpolation.)
    escaped = deny.replace("\\", "\\\\").replace('"', '\\"')
    profile = f'(version 1)(allow default)(deny file-read* (subpath "{escaped}"))'
    return ["sandbox-exec", "-p", profile, *command]
