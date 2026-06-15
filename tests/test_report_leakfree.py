"""Guard the one benchmark artifact that ships publicly.

``results/<run>/report.html`` is the published evaluation report. It is aggregate
by construction: it embeds a ``window.RUN`` payload that must carry only per-model
alpha and a normalized cost/speed index — never per-``(agent, task)`` rows, the
held-out task names, absolute cost/latency, or a path from the authoring machine.

This test pins the embedded payload to an allowlist (only the expected top-level
and per-agent keys survive) and scans the raw HTML for leak markers, so
regenerating the report can never silently reintroduce a leak.

The held-out frontier task names are themselves secret, so this file must not
commit them. ``test_report_excludes_held_out_slugs`` loads the real slugs at
runtime from ``HEIST_FRONTIER_TASKS_DIR`` (the private ``tasks/frontier``
directory) when it is set, and is skipped otherwise.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

# The only keys allowed in the embedded window.RUN payload.
ALLOWED_TOP_KEYS = {"agents", "rank_order_ids"}
ALLOWED_AGENT_KEYS = {"id", "label", "color", "alpha", "cost", "lat"}

# Substrings that would betray per-task results, absolute cost/time, the held-out
# test set, or the authoring machine. `cost_usd`/`latency_s`/`pairs`/`task` are the
# full report's per-row field names — their absence proves this is the aggregate cut.
LEAK_MARKERS = (
    "/Users/",
    "/home/",
    "hardbench",
    "adamthuvesen",
    "cost_usd",
    "latency_s",
    '"pairs"',
    '"tasks"',
    '"task":',
    '"checks"',
)


def _reports(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "results").glob("*/report.html"))


def _held_out_slugs() -> list[str]:
    """Real held-out frontier slugs, loaded from a private path at runtime.

    Committing these names to this public repo would itself leak the held-out
    set, so the list is never stored here. Point ``HEIST_FRONTIER_TASKS_DIR`` at
    the private ``tasks/frontier`` directory to enable the slug blocklist;
    without it the check is skipped and the allowlist/marker scans remain.
    """
    raw = os.environ.get("HEIST_FRONTIER_TASKS_DIR")
    if not raw:
        return []
    frontier = Path(raw)
    if not frontier.is_dir():
        return []
    return sorted(p.name for p in frontier.iterdir() if p.is_dir() and not p.name.startswith("__"))


def _embedded_run(html: str) -> dict:
    match = re.search(r"window\.RUN = (\{.*?\});", html)
    assert match, "could not find the embedded window.RUN payload"
    return json.loads(match.group(1))


def test_a_published_report_exists(repo_root: Path) -> None:
    # Without this, an accidental delete would make the guards below pass vacuously.
    assert _reports(repo_root), "expected a committed results/<run>/report.html"


def test_embedded_payload_has_only_aggregate_keys(repo_root: Path) -> None:
    for path in _reports(repo_root):
        run = _embedded_run(path.read_text())
        assert set(run) <= ALLOWED_TOP_KEYS, (
            f"{path}: window.RUN keys {sorted(run)} exceed {sorted(ALLOWED_TOP_KEYS)}"
        )
        agents = run.get("agents", [])
        assert agents, f"{path}: no agents in payload"
        for agent in agents:
            assert set(agent) <= ALLOWED_AGENT_KEYS, (
                f"{path}: agent keys {sorted(agent)} exceed {sorted(ALLOWED_AGENT_KEYS)}"
            )


def test_no_leak_markers_in_report(repo_root: Path) -> None:
    for path in _reports(repo_root):
        text = path.read_text()
        for marker in LEAK_MARKERS:
            assert marker not in text, f"{path} leaks marker {marker!r}"


def test_report_excludes_held_out_slugs(repo_root: Path) -> None:
    slugs = _held_out_slugs()
    if not slugs:
        pytest.skip(
            "set HEIST_FRONTIER_TASKS_DIR to the private frontier task set "
            "to enable the held-out slug blocklist"
        )
    for path in _reports(repo_root):
        text = path.read_text()
        for slug in slugs:
            assert slug not in text, f"{path} leaks held-out task slug {slug!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
