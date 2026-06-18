from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import heist.history as history_module
from heist.history import (
    BaselineRegistry,
    HistoryError,
    compare,
    cross_run_table,
    load_all_runs,
    load_run_results,
    resolve_run_ref,
)
from tests.fixtures.runs import (
    make_result,
    write_corrupt_run,
    write_synthetic_run,
    write_two_runs,
)


@pytest.fixture(autouse=True)
def _clear_results_cache() -> None:
    # The history module caches results.jsonl reads for the lifetime of the
    # process. Tests reuse run ids across tmp_path fixtures, so the cache
    # must not bleed between tests.
    history_module._cached_results.cache_clear()
    yield
    history_module._cached_results.cache_clear()


# ---------------------------------------------------------------------------
# load_all_runs / cross_run_table
# ---------------------------------------------------------------------------


def test_load_all_runs_empty_dir(tmp_path: Path) -> None:
    summaries, corrupt = load_all_runs(tmp_path)
    assert summaries == []
    assert corrupt == []


def test_load_all_runs_returns_descending_by_created_at(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    summaries, _ = load_all_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["run-b", "run-a"]
    assert summaries[0].mean_score == 0.5
    assert summaries[1].mean_score == 0.8


def test_load_all_runs_skips_dirs_without_manifest(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    (tmp_path / "no-manifest").mkdir()
    (tmp_path / "no-manifest" / "results.jsonl").write_text("")
    summaries, _ = load_all_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["ok"]


def test_load_all_runs_surfaces_corrupt_when_requested(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    write_corrupt_run(tmp_path, "broken")
    summaries_silent, corrupt_silent = load_all_runs(tmp_path)
    assert [s.run_id for s in summaries_silent] == ["ok"]
    assert corrupt_silent == []

    summaries, corrupt = load_all_runs(tmp_path, include_corrupt=True)
    assert [s.run_id for s in summaries] == ["ok"]
    assert [c.run_id for c in corrupt] == ["broken"]
    assert corrupt[0].reason


def test_cross_run_table_filters_by_agent_and_task(tmp_path: Path) -> None:
    older = datetime.now(UTC) - timedelta(hours=1)
    newer = datetime.now(UTC)
    write_synthetic_run(
        tmp_path,
        "r1",
        results=[
            make_result(run_id="r1", agent_id="a", task_id="t1", score=0.7),
            make_result(run_id="r1", agent_id="b", task_id="t2", score=0.4),
        ],
        created_at=older,
    )
    write_synthetic_run(
        tmp_path,
        "r2",
        results=[
            make_result(run_id="r2", agent_id="a", task_id="t1", score=0.9),
        ],
        created_at=newer,
    )

    rows_all = cross_run_table(tmp_path)
    # Sorted by created_at ascending.
    assert [r["run_id"] for r in rows_all] == ["r1", "r1", "r2"]

    rows_agent = cross_run_table(tmp_path, agent_id="a")
    assert {row["run_id"] for row in rows_agent} == {"r1", "r2"}
    assert all(row["agent_id"] == "a" for row in rows_agent)

    rows_task = cross_run_table(tmp_path, task_id="t2")
    assert [row["run_id"] for row in rows_task] == ["r1"]
    assert rows_task[0]["agent_id"] == "b"

    rows_pair = cross_run_table(tmp_path, agent_id="a", task_id="t1")
    assert len(rows_pair) == 2
    assert {row["run_id"] for row in rows_pair} == {"r1", "r2"}


def test_load_run_results_is_cached(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "r1")
    first = load_run_results(tmp_path, "r1")
    second = load_run_results(tmp_path, "r1")
    # Cache returns identical underlying data without re-reading disk;
    # equality is enough — we don't assert identity since model_copy paths
    # produce fresh objects.
    assert [r.run_id for r in first] == [r.run_id for r in second] == ["r1"]


# ---------------------------------------------------------------------------
# BaselineRegistry
# ---------------------------------------------------------------------------


def test_baseline_registry_set_and_list(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    registry = BaselineRegistry.load(tmp_path)
    assert registry.list() == {}
    assert registry.set(tmp_path, "ok", "v1") is None
    assert registry.list() == {"v1": "ok"}
    # Persisted to disk.
    reloaded = BaselineRegistry.load(tmp_path)
    assert reloaded.list() == {"v1": "ok"}


def test_baseline_registry_reassign_returns_previous(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok-1")
    write_synthetic_run(tmp_path, "ok-2")
    registry = BaselineRegistry.load(tmp_path)
    registry.set(tmp_path, "ok-1", "v1")
    previous = registry.set(tmp_path, "ok-2", "v1")
    assert previous == "ok-1"
    assert registry.list() == {"v1": "ok-2"}


def test_baseline_registry_unset(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    registry = BaselineRegistry.load(tmp_path)
    registry.set(tmp_path, "ok", "v1")
    removed = registry.unset(tmp_path, "v1")
    assert removed == "ok"
    assert registry.list() == {}
    with pytest.raises(HistoryError, match="not defined"):
        registry.unset(tmp_path, "v1")


def test_baseline_registry_rejects_reserved_names(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    registry = BaselineRegistry.load(tmp_path)
    for reserved in ("latest", "previous"):
        with pytest.raises(HistoryError, match="reserved"):
            registry.set(tmp_path, "ok", reserved)


def test_baseline_registry_rejects_unknown_run(tmp_path: Path) -> None:
    registry = BaselineRegistry.load(tmp_path)
    with pytest.raises(HistoryError, match="not found"):
        registry.set(tmp_path, "missing", "v1")


def test_baseline_registry_rejects_path_like_run_ids(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    registry = BaselineRegistry.load(tmp_path)

    with pytest.raises(HistoryError, match="invalid run id"):
        registry.set(tmp_path, "../outside-run", "escape")


def test_baseline_registry_corrupt_file_raises(tmp_path: Path) -> None:
    (tmp_path / "baselines.json").write_text('["not", "object"]')
    with pytest.raises(HistoryError, match="not a JSON object"):
        BaselineRegistry.load(tmp_path)


def test_baseline_registry_load_rejects_invalid_run_id(tmp_path: Path) -> None:
    # A hand-edited baselines.json pointing a tag at a path-like value must be
    # rejected at load, naming the file — not deferred to resolution time.
    (tmp_path / "baselines.json").write_text('{"v1": "../../etc/passwd"}')
    with pytest.raises(HistoryError, match="invalid run id"):
        BaselineRegistry.load(tmp_path)


def test_baseline_registry_load_rejects_reserved_tag(tmp_path: Path) -> None:
    (tmp_path / "baselines.json").write_text('{"latest": "some-run"}')
    with pytest.raises(HistoryError, match="reserved"):
        BaselineRegistry.load(tmp_path)


def test_load_all_runs_tolerates_naive_created_at(tmp_path: Path) -> None:
    # A manifest with a naive (tz-less) created_at must not crash the sort in
    # load_all_runs by comparing naive vs aware datetimes.
    write_synthetic_run(tmp_path, "aware-run", created_at=datetime.now(UTC))
    naive_dir = write_synthetic_run(tmp_path, "naive-run")
    manifest_path = naive_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["created_at"] = "2026-01-01T00:00:00"  # no offset → naive
    manifest_path.write_text(json.dumps(payload))

    summaries, _corrupt = load_all_runs(tmp_path)

    run_ids = {s.run_id for s in summaries}
    assert {"aware-run", "naive-run"} <= run_ids


# ---------------------------------------------------------------------------
# resolve_run_ref
# ---------------------------------------------------------------------------


def test_resolve_run_ref_literal_pass_through(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "concrete")
    run_id, banner = resolve_run_ref(tmp_path, "concrete")
    assert run_id == "concrete"
    assert banner is None


def test_resolve_run_ref_known_tag(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "target")
    registry = BaselineRegistry.load(tmp_path)
    registry.set(tmp_path, "target", "v1")
    run_id, banner = resolve_run_ref(tmp_path, "v1")
    assert run_id == "target"
    assert banner is None


def test_resolve_run_ref_latest_and_previous(tmp_path: Path) -> None:
    older, newer = write_two_runs(tmp_path)
    run_latest, _ = resolve_run_ref(tmp_path, "latest")
    run_prev, _ = resolve_run_ref(tmp_path, "previous")
    assert run_latest == newer
    assert run_prev == older


def test_resolve_run_ref_previous_with_only_one_run(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    with pytest.raises(HistoryError, match="only 1 run"):
        resolve_run_ref(tmp_path, "previous")


def test_resolve_run_ref_unknown_value(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "exists")
    with pytest.raises(HistoryError, match="could not resolve"):
        resolve_run_ref(tmp_path, "ghost")


def test_resolve_run_ref_ambiguous_tag_and_dir(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "real-target")
    write_synthetic_run(tmp_path, "shadow")
    registry = BaselineRegistry.load(tmp_path)
    registry.set(tmp_path, "real-target", "shadow")  # tag named 'shadow'
    run_id, banner = resolve_run_ref(tmp_path, "shadow")
    assert run_id == "real-target"
    assert banner is not None and "literal" in banner

    run_literal, banner_literal = resolve_run_ref(tmp_path, "shadow", literal=True)
    assert run_literal == "shadow"
    assert banner_literal is None


def test_resolve_run_ref_literal_validates_existence(tmp_path: Path) -> None:
    with pytest.raises(HistoryError, match="not found"):
        resolve_run_ref(tmp_path, "missing", literal=True)


def test_resolve_run_ref_rejects_path_like_literal_refs(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    with pytest.raises(HistoryError, match="invalid run id"):
        resolve_run_ref(tmp_path, "../ok", literal=True)


def test_resolve_run_ref_rejects_corrupt_baseline_targets(tmp_path: Path) -> None:
    # A path-like baseline target is now rejected when baselines.json is loaded
    # (inside resolve_run_ref), naming the file, rather than deferred to lookup.
    (tmp_path / "baselines.json").write_text('{"v1": "../outside"}')
    with pytest.raises(HistoryError, match="invalid run id"):
        resolve_run_ref(tmp_path, "v1")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_emits_score_and_cost_deltas(tmp_path: Path) -> None:
    write_two_runs(tmp_path)
    report = compare(tmp_path, "run-a", "run-b")
    assert len(report.rows) == 1
    [row] = report.rows
    assert row.score_a == 0.8
    assert row.score_b == 0.5
    assert row.delta_score == pytest.approx(-0.3)
    assert row.delta_cost_usd == pytest.approx(0.10)
    assert row.delta_latency_s == pytest.approx(1.0)
    assert row.regression == "score_drop"


def test_compare_classifies_pass_to_fail_regression(tmp_path: Path) -> None:
    write_synthetic_run(
        tmp_path,
        "pass",
        results=[
            make_result(run_id="pass", score=1.0, success=True),
        ],
    )
    write_synthetic_run(
        tmp_path,
        "fail",
        results=[
            make_result(
                run_id="fail",
                score=0.95,  # under threshold, but transitioned pass → fail
                success=False,
            ),
        ],
    )
    report = compare(tmp_path, "pass", "fail")
    [row] = report.rows
    assert row.regression == "pass_to_fail"


def test_compare_surfaces_task_additions_and_removals(tmp_path: Path) -> None:
    write_synthetic_run(
        tmp_path,
        "a",
        results=[
            make_result(run_id="a", task_id="shared"),
            make_result(run_id="a", task_id="only-a"),
        ],
    )
    write_synthetic_run(
        tmp_path,
        "b",
        results=[
            make_result(run_id="b", task_id="shared"),
            make_result(run_id="b", task_id="only-b"),
        ],
    )
    report = compare(tmp_path, "a", "b")
    assert [row.task_id for row in report.rows] == ["shared"]
    assert report.tasks_only_in_a == ["only-a"]
    assert report.tasks_only_in_b == ["only-b"]


def test_compare_harness_drift_banner(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "a", harness_git_sha="a" * 40, results=[make_result(run_id="a")])
    write_synthetic_run(tmp_path, "b", harness_git_sha="b" * 40, results=[make_result(run_id="b")])
    report = compare(tmp_path, "a", "b")
    assert report.harness_drift is not None
    assert "drift" in report.harness_drift


def test_compare_harness_drift_unknown_when_sha_missing(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "a", harness_git_sha=None, results=[make_result(run_id="a")])
    write_synthetic_run(tmp_path, "b", harness_git_sha="b" * 40, results=[make_result(run_id="b")])
    report = compare(tmp_path, "a", "b")
    assert report.harness_drift is not None
    assert "unknown" in report.harness_drift


def test_compare_self_yields_zero_deltas(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "only")
    report = compare(tmp_path, "only", "only")
    assert all(row.delta_score == 0.0 for row in report.rows)
    assert all(row.delta_cost_usd == 0.0 for row in report.rows)


def test_compare_rejects_unknown_run(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "ok")
    with pytest.raises(HistoryError, match="not found"):
        compare(tmp_path, "ok", "missing")
