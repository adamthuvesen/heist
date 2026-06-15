from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import heist.history as history_module
from heist.cli import app
from heist.history import compare
from heist.reporting import render_html, write_report
from tests.fixtures.runs import make_result, write_synthetic_run, write_two_runs

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_results_cache() -> None:
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


def test_render_html_baseline_section_absent_by_default(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    results = [make_result(run_id="only")]
    html = render_html(results)
    # Section opening marker only appears when a comparison is provided.
    assert '<section class="reveal baseline-section">' not in html
    assert "Versus baseline" not in html
    # And the empty token must not leak into the output.
    assert "{{BASELINE_SECTION}}" not in html


def test_render_html_baseline_section_renders_when_supplied(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    report = compare(tmp_path, "run-a", "run-b")
    # The "current" side feeds render_html's results argument.
    results_b = history_module.load_run_results(tmp_path, "run-b")
    html = render_html(results_b, baseline_comparison=report)
    assert '<section class="reveal baseline-section">' in html
    assert "Versus baseline" in html
    # Score-drop regression in the fixture (0.8 → 0.5) must be highlighted.
    assert 'class="regression"' in html
    assert "score drop" in html
    # Harness drift banner (a* vs b*) must render.
    assert "harness drift" in html or "drift" in html


def test_render_html_baseline_section_marks_pass_to_fail(tmp_path: Path) -> None:
    write_synthetic_run(
        tmp_path,
        "pass",
        results=[make_result(run_id="pass", score=1.0, success=True)],
    )
    write_synthetic_run(
        tmp_path,
        "fail",
        results=[
            make_result(run_id="fail", score=0.95, success=False),
        ],
    )
    report = compare(tmp_path, "pass", "fail")
    results = history_module.load_run_results(tmp_path, "fail")
    html = render_html(results, baseline_comparison=report)
    assert "pass → fail" in html


def test_write_report_includes_baseline_when_supplied(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    report = compare(tmp_path, "run-a", "run-b")
    results = history_module.load_run_results(tmp_path, "run-b")
    write_report(tmp_path / "run-b", results, baseline_comparison=report)
    written = (tmp_path / "run-b" / "report.html").read_text()
    assert "baseline-section" in written


def test_report_command_with_compare_baseline_flag(tmp_path: Path) -> None:
    older, newer = write_two_runs(tmp_path)
    result = runner.invoke(
        app,
        [
            "report",
            "--run",
            str(tmp_path / newer),
            "--compare-baseline",
            older,
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    html = (tmp_path / newer / "report.html").read_text()
    assert '<section class="reveal baseline-section">' in html


def test_report_command_compare_baseline_unknown_ref_exits(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    result = runner.invoke(
        app,
        [
            "report",
            "--run",
            str(tmp_path / "only"),
            "--compare-baseline",
            "ghost",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "could not resolve" in result.output


def test_report_without_baseline_does_not_render_section(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    result = runner.invoke(
        app,
        ["report", "--run", str(tmp_path / "only")],
    )
    assert result.exit_code == 0, result.output
    html = (tmp_path / "only" / "report.html").read_text()
    assert '<section class="reveal baseline-section">' not in html
    assert "Versus baseline" not in html
    assert "{{BASELINE_SECTION}}" not in html
