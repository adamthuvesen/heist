from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from heist.models import TaskDefinition, TaskSpec
from heist.tasks import (
    apply_reference,
    copy_workspace,
    load_tasks,
    run_hidden_grader,
    run_visible_tests,
)
from tests.fixtures.marker import write_marker_task


def test_examples_suite_has_three_tasks(repo_root: Path) -> None:
    tasks = load_tasks("examples", repo_root=repo_root)
    assert [task.id for task in tasks] == [
        "course-prereq-scheduler",
        "notification-routing",
        "ticket-lifecycle",
    ]
    assert all(task.spec.difficulty == "example" for task in tasks)


def test_task_yaml_id_must_match_directory_name(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "directory-name")
    task_yaml = tmp_path / "tasks" / "smoke" / "directory-name" / "task.yaml"
    task_yaml.write_text(task_yaml.read_text().replace("id: directory-name", "id: yaml-id"))

    with pytest.raises(ValueError, match="must match directory name"):
        load_tasks("smoke", repo_root=tmp_path)


def test_task_yaml_rejects_invalid_yaml(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "broken-yaml")
    task_yaml = tmp_path / "tasks" / "smoke" / "broken-yaml" / "task.yaml"
    task_yaml.write_text("id: [\n")

    with pytest.raises(ValueError, match="Task file contains invalid YAML"):
        load_tasks("smoke", repo_root=tmp_path)


def test_seed_workspaces_pass_visible_tests_but_fail_hidden_graders(
    tmp_path: Path, repo_root: Path
) -> None:
    # Seeds must score below the hardness ceiling: a seed drifting to ~0.95 would
    # silently weaken the 'success requires >= 0.999' contract.
    for task in load_tasks("examples", repo_root=repo_root):
        workspace = tmp_path / "seed" / task.id
        copy_workspace(task, workspace)
        visible = run_visible_tests(task, workspace)
        assert visible.returncode == 0, visible.stdout + visible.stderr
        hidden = run_hidden_grader(task, workspace)
        assert 0.0 <= hidden.score <= 0.7, (task.id, hidden.score)


def test_reference_solutions_pass_hidden_graders(tmp_path: Path, repo_root: Path) -> None:
    for task in load_tasks("examples", repo_root=repo_root):
        workspace = tmp_path / "reference" / task.id
        copy_workspace(task, workspace)
        apply_reference(task, workspace)
        hidden = run_hidden_grader(task, workspace)
        assert hidden.score == 1.0, task.id


def test_run_hidden_grader_respects_taskspec_grader_timeout(tmp_path: Path) -> None:
    task_dir = tmp_path / "slow"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "reference").mkdir()
    hidden_dir = task_dir / "hidden"
    hidden_dir.mkdir()
    (hidden_dir / "grader.py").write_text(
        'import time, sys\ntime.sleep(5)\nprint(\'{"score":1.0,"passed":true,"checks":[]}\')\n'
    )

    task = TaskDefinition(
        suite="smoke",
        spec=TaskSpec(
            id="slow-grader",
            title="Slow grader",
            category="fake",
            prompt="x",
            grader_timeout_s=1,
        ),
        path=task_dir,
        workspace_path=task_dir / "workspace",
        hidden_path=hidden_dir,
        reference_path=task_dir / "reference",
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_hidden_grader(task, task_dir / "workspace", timeout_s=60)


def test_copy_workspace_ignores_dot_git_pycache_pytest_cache_pyc(tmp_path: Path) -> None:
    # Each ignore pattern matters: stale __pycache__ from a previous agent run
    # would leak into the next workspace; .git would let a clever agent rewrite
    # baseline state; *.pyc bypasses the cache directories.
    source = tmp_path / "src-task"
    workspace = source / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "answer.txt").write_text("hi\n")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "foo.pyc").write_text("stale\n")
    (workspace / ".pytest_cache").mkdir()
    (workspace / ".pytest_cache" / "v").write_text("stale\n")
    (workspace / "stray.pyc").write_text("stale\n")
    git_dir = workspace / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (source / "hidden").mkdir()
    (source / "reference").mkdir()

    task = TaskDefinition(
        suite="smoke",
        spec=TaskSpec(
            id="ignore-git",
            title="Ignore .git",
            category="fake",
            prompt="x",
        ),
        path=source,
        workspace_path=workspace,
        hidden_path=source / "hidden",
        reference_path=source / "reference",
    )

    destination = tmp_path / "workspaces" / "copy"
    copy_workspace(task, destination)

    assert (destination / "src" / "answer.txt").exists()
    assert not (destination / ".git").exists()
    assert not (destination / "__pycache__").exists()
    assert not (destination / ".pytest_cache").exists()
    assert not (destination / "stray.pyc").exists()
