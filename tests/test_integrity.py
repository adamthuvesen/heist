from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from heist.integrity import detect_grader_access, require_sandbox_supported, sandbox_wrap
from heist.tasks import load_tasks
from tests.fixtures.marker import write_marker_task


def _marker_task(tmp_path: Path):
    write_marker_task(tmp_path)
    return load_tasks("smoke", repo_root=tmp_path)[0]


def test_detect_flags_successful_hidden_grader_read(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(f"running python {task.hidden_path}/grader.py to peek\n")
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("")
    access = detect_grader_access(stdout, stderr, task)
    assert access.contaminated is not None
    assert "hidden grader" in access.contaminated
    assert access.attempted is None


def test_detect_flags_successful_reference_read_in_stderr(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    (tmp_path / "stdout.txt").write_text("")
    stderr = tmp_path / "stderr.txt"
    stderr.write_text(f"cat {task.reference_path}/answer.txt\n")
    access = detect_grader_access(tmp_path / "stdout.txt", stderr, task)
    assert access.contaminated is not None
    assert "reference solution" in access.contaminated


def test_sandboxed_path_mention_is_attempt_not_contamination(tmp_path: Path) -> None:
    # Under a sandbox the read could not have succeeded, so a path mention is a
    # thwarted attempt: the score stays valid, but the attempt is recorded.
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(
        f"PermissionError: [Errno 1] Operation not permitted: "
        f"'{task.reference_path}/src/ingest.py'\n"
    )
    (tmp_path / "stderr.txt").write_text("")
    access = detect_grader_access(stdout, tmp_path / "stderr.txt", task, sandboxed=True)
    assert access.contaminated is None
    assert access.attempted is not None
    assert "reference solution" in access.attempted


def test_unsandboxed_path_mention_contaminates(tmp_path: Path) -> None:
    # No sandbox means the read went through — the run is invalid regardless of
    # how the path was phrased.
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(f"cat {task.hidden_path}/grader.py\nprint(score)\n")
    (tmp_path / "stderr.txt").write_text("")
    access = detect_grader_access(stdout, tmp_path / "stderr.txt", task, sandboxed=False)
    assert access.contaminated is not None
    assert access.attempted is None


def test_detect_flags_relative_grader_path(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("cd into the task dir, then: cat hidden/grader.py\n")
    (tmp_path / "stderr.txt").write_text("")
    access = detect_grader_access(stdout, tmp_path / "stderr.txt", task)
    assert access.contaminated is not None
    assert "hidden grader" in access.contaminated


def test_detect_flags_task_qualified_relative_path(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(f"opened {task.id}/reference/answer.txt\n")
    (tmp_path / "stderr.txt").write_text("")
    access = detect_grader_access(stdout, tmp_path / "stderr.txt", task)
    assert access.contaminated is not None
    assert "reference solution" in access.contaminated


def test_detect_does_not_flag_bare_dir_words(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("I kept the helper hidden and added a reference comment.\n")
    (tmp_path / "stderr.txt").write_text("")
    access = detect_grader_access(stdout, tmp_path / "stderr.txt", task)
    assert access.contaminated is None
    assert access.attempted is None


def test_detect_clean_transcript_passes(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("Editing answer.txt to write yes. Ran the visible tests. Done.\n")
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("ok\n")
    access = detect_grader_access(stdout, stderr, task)
    assert access.contaminated is None
    assert access.attempted is None


def test_detect_missing_or_empty_files_pass(tmp_path: Path) -> None:
    task = _marker_task(tmp_path)
    access = detect_grader_access(tmp_path / "absent.txt", None, task)
    assert access.contaminated is None
    assert access.attempted is None


def test_sandbox_wrap_prepends_deny_profile(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    command = ["cursor-agent", "-p", "{prompt-already-substituted}"]
    wrapped = sandbox_wrap(command, tasks_root)
    assert wrapped[0] == "sandbox-exec"
    assert wrapped[1] == "-p"
    assert "deny file-read*" in wrapped[2]
    assert os.path.realpath(tasks_root) in wrapped[2]
    assert wrapped[3:] == command


def test_sandbox_wrap_leaves_empty_command_unchanged() -> None:
    assert sandbox_wrap([], Path("/repo/tasks")) == []


def test_require_sandbox_supported_off_macos_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        require_sandbox_supported()


def test_require_sandbox_supported_on_macos_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    require_sandbox_supported()
