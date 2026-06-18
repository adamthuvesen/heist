from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from heist.cli import app
from heist.runner import load_results, run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task
from tests.fixtures.runs import make_result, write_synthetic_run

runner = CliRunner()


def _build_run(tmp_path: Path, run_id: str = "cli-grade-run") -> Path:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, _ = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id=run_id,
    )
    return Path(manifest.run_dir)


def _assert_cli_error(result: Result, message: str) -> None:
    assert result.exit_code != 0
    assert message in result.output
    assert "Traceback" not in result.output


def test_grade_command_uses_load_manifest(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    result = runner.invoke(
        app,
        ["grade", "--run", str(run_dir), "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Regraded 1 rows" in result.output
    assert (run_dir / "regrade-results.jsonl").exists()
    assert (run_dir / "regrade-summary.md").exists()
    assert len(load_results(run_dir)) == 1


def test_grade_command_reports_stale_task_rows_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "marker")
    run_dir = write_synthetic_run(
        tmp_path / "runs",
        "stale-run",
        results=[make_result(run_id="stale-run", task_id="ghost-task")],
    )

    result = runner.invoke(
        app,
        ["grade", "--run", str(run_dir), "--repo-root", str(tmp_path)],
    )

    _assert_cli_error(result, "references task 'ghost-task'")


def test_tasks_list_command_shows_all_smoke_ids(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "alpha")
    write_marker_task(tmp_path, "beta")

    result = runner.invoke(
        app,
        ["tasks", "list", "--suite", "smoke", "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output


def test_report_command_prints_summary(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, run_id="cli-report-run")

    result = runner.invoke(app, ["report", "--run", str(run_dir)])

    assert result.exit_code == 0, result.output
    # Header alone isn't enough — a regression that renders the header but
    # drops the body would still pass. Assert the agent label and the alpha
    # ranking chart appear so the body is genuinely populated.
    assert "HEIST Run Report" in result.output
    assert "Fake pass" in result.output
    assert "## alpha Ranking" in result.output


def test_suites_list_command_includes_smoke(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "marker")

    result = runner.invoke(app, ["suites", "list", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "smoke" in result.output


def test_agents_list_command_lists_default_registry(tmp_path: Path) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(app, ["agents", "list", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    # Rich's table renderer may wrap or truncate the rendered cells, so match
    # on substrings that always survive (model ids, not the agent id chrome).
    assert "claude-opus-4-8" in result.output
    assert "gpt-5.5" in result.output


def test_config_init_writes_starter_toml(tmp_path: Path) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(app, ["config", "init", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    cfg_path = tmp_path / "heist.toml"
    assert cfg_path.exists()
    text = cfg_path.read_text()
    assert "[defaults]" in text
    assert "suite" in text


def test_config_show_roundtrips_init_output(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    runner.invoke(app, ["config", "init", "--repo-root", str(tmp_path)])

    result = runner.invoke(app, ["config", "show", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    # Roundtrip: `show` must surface fields that `init` wrote to heist.toml.
    # Asserting key default keys catches a regression where `show` prints
    # the header but no values, or reads a stale file.
    assert "[defaults]" in result.output
    assert "suite" in result.output
    assert "jobs" in result.output


def test_config_show_reports_invalid_env_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(
        app,
        ["config", "show", "--repo-root", str(tmp_path)],
        env={"HEIST_PROGRESS": "maybe"},
    )

    _assert_cli_error(result, "progress must be a boolean")


@pytest.mark.parametrize(
    "flag,value,message",
    [
        ("--jobs", "0", "jobs must be >= 1"),
        ("--timeout", "0", "timeout must be >= 1"),
        ("--retry", "-1", "retry must be >= 0"),
        ("--provider-jobs", "claude=0", "must be >= 1"),
    ],
)
def test_run_reports_invalid_numeric_flags_without_traceback(
    tmp_path: Path, flag: str, value: str, message: str
) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "smoke",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            flag,
            value,
        ],
    )

    _assert_cli_error(result, message)


def test_run_reports_path_like_run_id_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "smoke",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--run-id",
            "../outside",
        ],
    )

    _assert_cli_error(result, "invalid run id")


def test_tasks_list_reports_invalid_task_metadata_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "directory-name")
    task_yaml = tmp_path / "tasks" / "smoke" / "directory-name" / "task.yaml"
    task_yaml.write_text(task_yaml.read_text().replace("id: directory-name", "id: yaml-id"))

    result = runner.invoke(
        app,
        ["tasks", "list", "--suite", "smoke", "--repo-root", str(tmp_path)],
    )

    _assert_cli_error(result, "must match directory name")


def test_run_reports_unknown_agent_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "smoke",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--agent",
            "does-not-exist",
        ],
    )

    _assert_cli_error(result, "Unknown agent")


def test_agents_list_reports_invalid_agent_file_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    agent_file = tmp_path / "agents.yaml"
    agent_file.write_text("- not\n- a\n- mapping\n")

    result = runner.invoke(
        app,
        [
            "agents",
            "list",
            "--repo-root",
            str(tmp_path),
            "--agent-file",
            str(agent_file),
        ],
    )

    _assert_cli_error(result, "Agent file must contain a mapping")


def test_agents_list_reports_malformed_agent_file_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    agent_file = tmp_path / "agents.yaml"
    agent_file.write_text("agents: [\n")

    result = runner.invoke(
        app,
        [
            "agents",
            "list",
            "--repo-root",
            str(tmp_path),
            "--agent-file",
            str(agent_file),
        ],
    )

    _assert_cli_error(result, "Agent file contains invalid YAML")


def test_tasks_list_reports_malformed_task_yaml_without_traceback(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "broken-yaml")
    task_yaml = tmp_path / "tasks" / "smoke" / "broken-yaml" / "task.yaml"
    task_yaml.write_text("id: [\n")

    result = runner.invoke(
        app,
        ["tasks", "list", "--suite", "smoke", "--repo-root", str(tmp_path)],
    )

    _assert_cli_error(result, "Task file contains invalid YAML")


@pytest.mark.parametrize(
    "args,message",
    [
        (["report", "--run"], "results.jsonl"),
        (["grade", "--run"], "manifest.json"),
        (["export", "eval-audit", "--run"], "results.jsonl"),
    ],
)
def test_run_artifact_commands_report_missing_run_without_traceback(
    tmp_path: Path, args: list[str], message: str
) -> None:
    result = runner.invoke(app, [*args, str(tmp_path / "missing-run")])

    _assert_cli_error(result, message)


def test_report_command_reports_corrupt_results_without_traceback(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken-run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("{not json}\n")

    result = runner.invoke(app, ["report", "--run", str(run_dir)])

    _assert_cli_error(result, "Expecting property name")
