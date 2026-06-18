from __future__ import annotations

from pathlib import Path

import pytest

from heist.agents import DEFAULT_AGENTS, resolve_agents
from heist.tasks import list_suites, select_tasks
from tests.fixtures.marker import write_marker_task


def test_resolve_agents_by_provider_returns_all_matching() -> None:
    selected = resolve_agents(agent_ids=None, providers=["claude"])
    assert {agent.id for agent in selected} == {
        agent_id for agent_id, spec in DEFAULT_AGENTS.items() if spec.provider == "claude"
    }


def test_resolve_agents_all_agents_returns_full_registry() -> None:
    selected = resolve_agents(agent_ids=None, all_agents=True)
    assert {agent.id for agent in selected} == set(DEFAULT_AGENTS)


def test_resolve_agents_exclude_drops_specific_agent() -> None:
    selected = resolve_agents(agent_ids=None, all_agents=True, exclude=["cursor-grok-4.3"])
    assert "cursor-grok-4.3" not in {agent.id for agent in selected}


def test_resolve_agents_exclude_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown agent"):
        resolve_agents(agent_ids=None, all_agents=True, exclude=["does-not-exist"])


def test_resolve_agents_uses_default_set_when_no_explicit_selection() -> None:
    selected = resolve_agents(
        agent_ids=None,
        default_set=["claude-opus-4.8-high", "codex-gpt-5.5-xhigh"],
    )
    assert [agent.id for agent in selected] == [
        "claude-opus-4.8-high",
        "codex-gpt-5.5-xhigh",
    ]


def test_resolve_agents_explicit_ids_beat_default_set() -> None:
    selected = resolve_agents(
        agent_ids=["cursor-composer-2.5"],
        default_set=["claude-opus-4.8-high"],
    )
    assert [agent.id for agent in selected] == ["cursor-composer-2.5"]


def test_select_tasks_glob_filters_ids(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "ledger-a")
    write_marker_task(tmp_path, "ledger-b")
    write_marker_task(tmp_path, "other")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path, glob="ledger-*")
    assert {task.id for task in tasks} == {"ledger-a", "ledger-b"}


def test_select_tasks_category_filters_when_present(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "alpha")
    tasks = select_tasks(suite="smoke", repo_root=tmp_path, category="fake")
    assert {task.id for task in tasks} == {"alpha"}


def test_select_tasks_glob_no_match_raises(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "alpha")
    with pytest.raises(KeyError, match="did not match"):
        select_tasks(suite="smoke", repo_root=tmp_path, glob="nope-*")


def test_list_suites_returns_directories_with_tasks(tmp_path: Path) -> None:
    write_marker_task(tmp_path, "alpha")
    (tmp_path / "tasks" / "empty").mkdir()
    assert list_suites(tmp_path) == ["smoke"]
