from __future__ import annotations

import sys
from pathlib import Path

import pytest

from heist.agents import DEFAULT_AGENTS, load_agent_file, resolve_agents


def test_default_agents_use_strong_models() -> None:
    codex = DEFAULT_AGENTS["codex-gpt-5.5-xhigh"]
    claude = DEFAULT_AGENTS["claude-opus-4.8-high"]
    cursor = DEFAULT_AGENTS["cursor-composer-2.5"]

    assert codex.command == [
        "codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        "gpt-5.5",
        "-c",
        "model_reasoning_effort=xhigh",
        "-",
    ]
    assert codex.prompt_via_stdin is True
    assert claude.model_id == "claude-opus-4-8"
    assert claude.command[claude.command.index("--effort") + 1] == "high"
    assert claude.required_env == []
    assert cursor.command[:3] == ["cursor-agent", "-p", "--output-format"]
    assert "--stream-partial-output" in cursor.command


def test_cursor_factory_emits_byte_identical_command() -> None:
    composer_25 = DEFAULT_AGENTS["cursor-composer-2.5"]
    assert composer_25.command[-2:] == ["composer-2.5", "{prompt}"]

    grok = DEFAULT_AGENTS["cursor-grok-4.3"]
    assert grok.command == [
        "cursor-agent",
        "-p",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--force",
        "--trust",
        "--model",
        "grok-4.3",
        "{prompt}",
    ]
    kimi = DEFAULT_AGENTS["cursor-kimi-k2.5"]
    assert kimi.command[-2:] == ["kimi-k2.5", "{prompt}"]
    gemini = DEFAULT_AGENTS["cursor-gemini-3.5-flash"]
    assert gemini.command[-2:] == ["gemini-3.5-flash", "{prompt}"]


def test_opencode_factory_emits_pty_wrapped_command() -> None:
    agent = DEFAULT_AGENTS["openrouter-deepseek-v4-pro"]
    assert agent.provider == "opencode"
    assert agent.model_id == "openrouter/deepseek/deepseek-v4-pro"
    assert agent.required_env == ["OPENROUTER_API_KEY"]
    assert agent.env_overrides == {
        "XDG_DATA_HOME": "{agent_home}/xdg-data",
        "XDG_CACHE_HOME": "{agent_home}/xdg-cache",
        "XDG_STATE_HOME": "{agent_home}/xdg-state",
    }
    assert agent.command == [
        sys.executable,
        "-m",
        "heist.pty_agent",
        "opencode",
        "run",
        "--format",
        "json",
        "--dir",
        "{workspace}",
        "--model",
        "openrouter/deepseek/deepseek-v4-pro",
        "{prompt}",
    ]


def test_opus_48_uses_high_effort_claude_command() -> None:
    opus_48 = DEFAULT_AGENTS["claude-opus-4.8-high"]
    assert opus_48.provider == "claude"
    assert opus_48.model_id == "claude-opus-4-8"
    # Bypass permissions, stream-json, high effort — the standard frontier
    # Claude invocation, only the model id changes between Claude agents.
    assert opus_48.command == [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        "claude-opus-4-8",
        "--effort",
        "high",
        "--output-format",
        "stream-json",
        "--verbose",
        "{prompt}",
    ]


def test_xhigh_agent_uses_maximum_effort_flag() -> None:
    codex = DEFAULT_AGENTS["codex-gpt-5.5-xhigh"]
    assert codex.command[codex.command.index("-c") + 1] == "model_reasoning_effort=xhigh"


DEFAULT_AGENT_ORDER = [
    "codex-gpt-5.5-xhigh",
    "codex-gpt-5.4-mini",
    "claude-opus-4.8-high",
    "claude-sonnet-4.6-high",
    "cursor-composer-2.5",
    "cursor-grok-4.3",
    "cursor-kimi-k2.5",
    "cursor-gemini-3.5-flash",
    "openrouter-deepseek-v4-pro",
    "openrouter-qwen3.7-max",
]


def test_resolve_agents_defaults_to_full_frontier_set() -> None:
    # Coverage: set membership. Adding/removing an agent must fail this test
    # (and force an explicit decision about whether the default registry should
    # change).
    assert {agent.id for agent in resolve_agents(None)} == set(DEFAULT_AGENT_ORDER)


def test_resolve_agents_default_order_drives_report_layout() -> None:
    # Coverage: display order. The HTML report and masthead chip list render
    # agents in this exact sequence (a deliberate ranking, not insertion order).
    # If you add an agent, decide where in the lineup it goes and update the
    # list above — don't let dict insertion order silently dictate the lineup.
    assert [agent.id for agent in resolve_agents(None)] == DEFAULT_AGENT_ORDER


def test_load_agent_file_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="Agent file must contain a mapping"):
        load_agent_file(path)


def test_load_agent_file_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text("agents: [\n")

    with pytest.raises(ValueError, match="Agent file contains invalid YAML"):
        load_agent_file(path)
