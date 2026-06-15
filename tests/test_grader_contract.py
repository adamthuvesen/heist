from __future__ import annotations

from pathlib import Path

import pytest

from heist.models import TaskDefinition, TaskSpec
from heist.tasks import GraderInvalidOutput, run_hidden_grader


def _task_with_grader(tmp_path: Path, grader_src: str) -> TaskDefinition:
    task_dir = tmp_path / "task"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    (task_dir / "reference").mkdir()
    hidden = task_dir / "hidden"
    hidden.mkdir()
    (hidden / "grader.py").write_text(grader_src)
    return TaskDefinition(
        suite="smoke",
        spec=TaskSpec(
            id="contract-task",
            title="Contract task",
            category="fake",
            prompt="x",
        ),
        path=task_dir,
        workspace_path=workspace,
        hidden_path=hidden,
        reference_path=task_dir / "reference",
    )


def test_grader_with_non_json_stdout_fails_loudly(tmp_path: Path) -> None:
    # AGENTS.md invariant: 'graders must fail loudly on invalid JSON / skipped
    # checks'. A grader that prints a non-JSON line must raise, not silently
    # produce a 0-score "pass".
    task = _task_with_grader(tmp_path, "print('definitely not json')\n")
    with pytest.raises(GraderInvalidOutput, match=r"unparseable JSON"):
        run_hidden_grader(task, task.workspace_path)


def test_grader_with_schema_violating_json_fails_loudly(tmp_path: Path) -> None:
    # score > 1.0 violates Field(ge=0, le=1). The grader must not produce a
    # GraderResult — pydantic must reject it.
    task = _task_with_grader(
        tmp_path,
        'print(\'{"score": 2.5, "passed": true, "checks": []}\')\n',
    )
    with pytest.raises(GraderInvalidOutput, match=r"unparseable JSON.*score|less_than_equal"):
        run_hidden_grader(task, task.workspace_path)


def test_grader_with_nonzero_exit_fails_loudly(tmp_path: Path) -> None:
    task = _task_with_grader(
        tmp_path,
        'import sys; print(\'{"score": 1.0, "passed": true, "checks": []}\'); sys.exit(2)\n',
    )
    with pytest.raises(RuntimeError, match=r"exit 2"):
        run_hidden_grader(task, task.workspace_path)


def test_grader_with_missing_required_field_fails_loudly(tmp_path: Path) -> None:
    # Missing 'checks' key — required by GraderResult.
    task = _task_with_grader(
        tmp_path,
        'print(\'{"score": 1.0, "passed": true}\')\n',
    )
    with pytest.raises(GraderInvalidOutput, match=r"checks"):
        run_hidden_grader(task, task.workspace_path)
