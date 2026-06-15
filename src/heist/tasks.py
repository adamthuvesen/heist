from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from heist.models import GraderResult, TaskDefinition, TaskSpec
from heist.paths import find_repo_root
from heist.subprocess_utils import run_subprocess_safely


def task_root(repo_root: Path | None = None) -> Path:
    return (repo_root or find_repo_root()) / "tasks"


def list_suites(repo_root: Path | None = None) -> list[str]:
    """Return suite names that contain at least one task directory."""
    root = task_root(repo_root)
    if not root.exists():
        return []
    suites: list[str] = []
    for suite in root.iterdir():
        if not suite.is_dir():
            continue
        if any(child.is_dir() and (child / "task.yaml").exists() for child in suite.iterdir()):
            suites.append(suite.name)
    return sorted(suites)


def load_tasks(suite: str = "smoke", repo_root: Path | None = None) -> list[TaskDefinition]:
    suite_dir = task_root(repo_root) / suite
    if not suite_dir.exists():
        raise FileNotFoundError(f"Unknown suite {suite!r}: {suite_dir}")

    tasks: list[TaskDefinition] = []
    seen_ids: set[str] = set()
    for task_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        spec_path = task_dir / "task.yaml"
        if not spec_path.exists():
            continue
        try:
            raw_spec = yaml.safe_load(spec_path.read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Task file contains invalid YAML: {spec_path}: {error}") from error
        spec = TaskSpec.model_validate(raw_spec)
        if spec.id != task_dir.name:
            raise ValueError(
                f"Task id mismatch in {spec_path}: id {spec.id!r} must match "
                f"directory name {task_dir.name!r}"
            )
        if spec.id in seen_ids:
            raise ValueError(f"Duplicate task id {spec.id!r} in suite {suite!r}")
        seen_ids.add(spec.id)
        workspace_path = task_dir / "workspace"
        hidden_path = task_dir / "hidden"
        reference_path = task_dir / "reference"
        for required in [workspace_path, hidden_path, reference_path]:
            if not required.exists():
                raise FileNotFoundError(f"Task {spec.id} missing {required.name}/")
        tasks.append(
            TaskDefinition(
                suite=suite,
                spec=spec,
                path=task_dir,
                workspace_path=workspace_path,
                hidden_path=hidden_path,
                reference_path=reference_path,
            )
        )
    return tasks


def select_tasks(
    suite: str = "smoke",
    task_ids: list[str] | None = None,
    repo_root: Path | None = None,
    *,
    glob: str | None = None,
    category: str | None = None,
) -> list[TaskDefinition]:
    """Select tasks from a suite.

    Filters compose: explicit `task_ids` (exact), then `glob` (fnmatch on id),
    then `category`. Returns the full suite when no filters are given.
    """
    tasks = load_tasks(suite=suite, repo_root=repo_root)

    if task_ids:
        by_id = {task.id: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise KeyError(f"Unknown task(s): {', '.join(missing)}")
        tasks = [by_id[task_id] for task_id in task_ids]

    if glob:
        tasks = [task for task in tasks if fnmatch.fnmatch(task.id, glob)]
        if not tasks:
            raise KeyError(f"--task-glob {glob!r} did not match any task in suite {suite!r}")

    if category:
        wanted = category.strip().lower()
        tasks = [task for task in tasks if task.spec.category.lower() == wanted]
        if not tasks:
            raise KeyError(
                f"--task-category {category!r} did not match any task in suite {suite!r}"
            )

    return tasks


def copy_workspace(task: TaskDefinition, destination: Path) -> None:
    if destination.exists():
        # Defensive: rmtree only when destination resolves to something safely
        # below a `workspaces/`/`seed/`/`reference/` dir — never up the tree.
        resolved = destination.resolve()
        guard_segments = {"workspaces", "seed", "reference", "seed-frontier", "regrade"}
        if not any(part in guard_segments for part in resolved.parts):
            raise RuntimeError(
                f"refusing to rmtree {resolved}: not inside a recognised heist workspace root"
            )
        shutil.rmtree(destination)
    shutil.copytree(
        task.workspace_path,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", ".git"),
    )


def apply_reference(task: TaskDefinition, workspace: Path) -> None:
    for source in task.reference_path.rglob("*"):
        if source.is_dir():
            continue
        target = workspace / source.relative_to(task.reference_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_visible_tests(
    task: TaskDefinition,
    workspace: Path,
    timeout_s: int = 60,
    *,
    pgid_registry: set[int] | None = None,
    pgid_lock: threading.Lock | None = None,
) -> subprocess.CompletedProcess[str]:
    result = run_subprocess_safely(
        list(task.spec.visible_test_command),
        cwd=workspace,
        timeout_s=timeout_s,
        pgid_registry=pgid_registry,
        pgid_lock=pgid_lock,
    )
    if result.timed_out:
        raise subprocess.TimeoutExpired(cmd=task.spec.visible_test_command, timeout=timeout_s)
    return subprocess.CompletedProcess(
        args=list(task.spec.visible_test_command),
        returncode=result.returncode or 0,
        stdout=result.stdout.decode(errors="replace"),
        stderr=result.stderr.decode(errors="replace"),
    )


class GraderInvalidOutput(RuntimeError):
    """Grader exited 0 but its stdout is not a parseable GraderResult.
    Distinct from a non-zero exit (grader code crashed) so the runner can
    label the row differently and consumers can triage faster."""


# Hard ceiling on a grader's JSON result line. Real graders emit a few KB at
# most; anything past this is a buggy grader streaming raw data into the
# payload slot, and validating it via Pydantic would consume RAM + skew
# timing reports.
_MAX_GRADER_PAYLOAD_BYTES = 256 * 1024


def _parse_grader_payload(stdout_text: str, task_id: str) -> GraderResult:
    # Contract: the grader's final non-empty line of stdout is the JSON result.
    # Earlier lines (debug prints, warnings) are tolerated. This is more
    # forgiving than "entire stdout must be JSON" while keeping the failure
    # mode loud when no JSON shows up at all.
    payload_line = ""
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if stripped:
            payload_line = stripped
    if not payload_line:
        raise GraderInvalidOutput(f"grader for {task_id} produced no output to parse")
    if len(payload_line) > _MAX_GRADER_PAYLOAD_BYTES:
        raise GraderInvalidOutput(
            f"grader for {task_id} payload is {len(payload_line)} bytes, "
            f"max {_MAX_GRADER_PAYLOAD_BYTES}"
        )
    try:
        return GraderResult.model_validate_json(payload_line)
    except Exception as error:
        raise GraderInvalidOutput(
            f"grader for {task_id} produced unparseable JSON: {error}"
        ) from error


def run_hidden_grader(
    task: TaskDefinition,
    workspace: Path,
    timeout_s: int = 60,
    *,
    pgid_registry: set[int] | None = None,
    pgid_lock: threading.Lock | None = None,
) -> GraderResult:
    effective_timeout = task.spec.grader_timeout_s or timeout_s
    result = run_subprocess_safely(
        [sys.executable, str(task.hidden_path / "grader.py"), str(workspace)],
        cwd=task.hidden_path,
        timeout_s=effective_timeout,
        pgid_registry=pgid_registry,
        pgid_lock=pgid_lock,
    )
    if result.timed_out:
        raise subprocess.TimeoutExpired(
            cmd=[sys.executable, str(task.hidden_path / "grader.py")],
            timeout=effective_timeout,
        )
    stdout_text = result.stdout.decode(errors="replace")
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        # Lead with the grader's last stderr line — that's typically the real
        # exception ("RuntimeError: ..."), not Python's "Traceback (most recent
        # call last):" framing. Falls back to a generic message if stderr is
        # empty.
        stderr_lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        actionable = stderr_lines[-1] if stderr_lines else f"exit {result.returncode}"
        raise RuntimeError(
            f"grader {task.id}: {actionable}\n"
            f"--- exit {result.returncode} ---\n"
            f"stdout:\n{stdout_text}\nstderr:\n{stderr_text}"
        )
    return _parse_grader_payload(stdout_text, task.id)
