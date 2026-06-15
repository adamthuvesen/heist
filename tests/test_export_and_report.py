from __future__ import annotations

from pathlib import Path

import polars as pl

from heist.export import export_eval_audit
from heist.reporting import render_markdown, summarize_by_agent
from heist.runner import load_results, run_benchmark
from heist.tasks import select_tasks
from tests.fixtures.marker import fake_agent, write_marker_task


def test_report_marks_saturation(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="report-run",
    )
    report = render_markdown(results)
    assert "SATURATED" in report
    assert "| Agent | Task | Category | Score | Status | Time | Cost | Cost source |" in report
    assert load_results(Path(manifest.run_dir))[0].success is True


def test_report_uses_primary_cost(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="report-cost-run",
    )

    report = render_markdown(results)
    assert "$1.2300" in report
    assert "| Fake reported | `marker` | fake | 100.0% | pass |" in report
    assert "| reported |" in report


def test_report_score_chart_sorts_highest_left(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("fail"), fake_agent("pass")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="chart-run",
    )

    report = render_markdown(results)

    assert "## Alpha (α) Ranking" in report
    assert "```text" in report
    chart_start = report.index("## Alpha (α) Ranking")
    chart_block = report[chart_start : report.index("## Hardness Gate", chart_start)]
    pass_idx = chart_block.index("Fake pass")
    fail_idx = chart_block.index("Fake fail")
    assert pass_idx < fail_idx, "highest-scoring agent must appear first (leftmost)"


def test_summary_splits_total_and_success_latency(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    _, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("pass"), fake_agent("fail")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="success-latency-run",
        jobs=1,
    )

    rows = {row["agent_id"]: row for row in summarize_by_agent(results)}

    pass_row = rows["fake-pass"]
    fail_row = rows["fake-fail"]

    # Passing agent: success_latency mirrors total_latency, median is the per-task value.
    assert pass_row["successes"] == 1
    assert pass_row["success_latency"] == pass_row["total_latency"] > 0
    assert pass_row["median_success_latency"] == pass_row["success_latency"]

    # Failing agent: still spent wall-clock time, but zero passing-task latency.
    assert fail_row["successes"] == 0
    assert fail_row["total_latency"] > 0
    assert fail_row["success_latency"] == 0.0
    assert fail_row["median_success_latency"] == 0.0

    report = render_markdown(results)
    assert "Time | Time (passed)" in report
    # Agent with no successes renders n/a in the Time (passed) column.
    fail_line = next(line for line in report.splitlines() if line.startswith("| Fake fail |"))
    assert fail_line.endswith(" n/a |")


def test_eval_audit_export_writes_canonical_rows(tmp_path: Path) -> None:
    write_marker_task(tmp_path)
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)
    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[fake_agent("reported", model_id="gpt-5.5")],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="export-run",
    )

    path = export_eval_audit(Path(manifest.run_dir), results)
    frame = pl.read_parquet(path)
    assert frame.select("agent_id").to_series().to_list() == ["fake-reported"]
    assert frame.select("harness").to_series().to_list() == ["heist"]
    assert frame.select("success").to_series().to_list() == [True]
    assert frame.select("cost_usd").to_series().to_list() == [1.23]
    assert frame.select("cost_source").to_series().to_list() == ["reported"]


def test_eval_audit_export_handles_multi_agent_multi_task_mixed_outcomes(
    tmp_path: Path,
) -> None:
    # Real-world shape: multiple agents × multiple tasks, with some passes,
    # some fails, and some errored rows. Single-row exports can't catch
    # ordering bugs or per-cell cost_source attribution regressions.
    write_marker_task(tmp_path, "marker-a")
    write_marker_task(tmp_path, "marker-b")
    # Break marker-b's grader to force an errored row.
    (tmp_path / "tasks" / "smoke" / "marker-b" / "hidden" / "grader.py").write_text(
        "raise RuntimeError('boom')\n"
    )
    tasks = select_tasks(suite="smoke", repo_root=tmp_path)

    manifest, results = run_benchmark(
        repo_root=tmp_path,
        suite="smoke",
        agents=[
            fake_agent("pass", model_id="gpt-5.5"),
            fake_agent("reported", model_id="gpt-5.5"),
        ],
        tasks=tasks,
        runs_dir=tmp_path / "runs",
        timeout_s=5,
        run_id="export-multi-run",
    )

    path = export_eval_audit(Path(manifest.run_dir), results)
    frame = pl.read_parquet(path)

    # 2 agents × 2 tasks = 4 rows, every (agent, task) pair present.
    assert frame.height == 4
    pairs = set(
        zip(
            frame["agent_id"].to_list(),
            frame["task_id"].to_list(),
            strict=True,
        )
    )
    assert pairs == {
        ("fake-pass", "marker-a"),
        ("fake-pass", "marker-b"),
        ("fake-reported", "marker-a"),
        ("fake-reported", "marker-b"),
    }

    by_pair = {(row["agent_id"], row["task_id"]): row for row in frame.iter_rows(named=True)}
    # marker-a runs land cleanly; marker-b errors out for both.
    assert by_pair[("fake-pass", "marker-a")]["success"] is True
    assert by_pair[("fake-pass", "marker-b")]["success"] is None
    assert by_pair[("fake-pass", "marker-b")]["outcome_status"] == "errored"
    assert by_pair[("fake-reported", "marker-a")]["cost_source"] == "reported"
    # Errored rows still carry reported cost when the agent emitted it —
    # fake-reported writes total_cost_usd before the grader is even run, so
    # the cost should survive the errored grader.
    assert by_pair[("fake-reported", "marker-b")]["cost_source"] == "reported"
