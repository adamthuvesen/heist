from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from heist.models import AgentSpec


def _opencode_agent(*, id: str, label: str, model_id: str, variant: str | None = None) -> AgentSpec:
    """Build an opencode agent routed through OpenRouter under a PTY shim."""
    command: list[str] = [
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
        model_id,
    ]
    if variant is not None:
        command.extend(["--variant", variant])
    command.append("{prompt}")
    return AgentSpec(
        id=id,
        label=label,
        provider="opencode",
        model_id=model_id,
        required_env=["OPENROUTER_API_KEY"],
        # Opencode stores sessions in a shared SQLite DB under XDG data home.
        # HEIST runs many tasks concurrently; isolate storage per job so
        # parallel `opencode run` processes do not hit "database is locked".
        env_overrides={
            "XDG_DATA_HOME": "{agent_home}/xdg-data",
            "XDG_CACHE_HOME": "{agent_home}/xdg-cache",
            "XDG_STATE_HOME": "{agent_home}/xdg-state",
        },
        command=command,
    )


def _cursor_agent(*, id: str, label: str, model_id: str) -> AgentSpec:
    """Build a Cursor agent that wraps `cursor-agent -p` for a given model."""
    return AgentSpec(
        id=id,
        label=label,
        provider="cursor",
        model_id=model_id,
        command=[
            "cursor-agent",
            "-p",
            "--output-format",
            "stream-json",
            "--stream-partial-output",
            "--force",
            "--trust",
            "--model",
            model_id,
            "{prompt}",
        ],
    )


DEFAULT_AGENTS: dict[str, AgentSpec] = {
    "codex-gpt-5.5-xhigh": AgentSpec(
        id="codex-gpt-5.5-xhigh",
        label="Codex GPT-5.5 XHigh",
        provider="codex",
        model_id="gpt-5.5",
        command=[
            "codex",
            "exec",
            "--json",
            # Skip ~/.codex/config.toml MCP servers — they can block startup for
            # minutes (or hang until timeout) in headless benchmark runs.
            "--ignore-user-config",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=xhigh",
            "-",
        ],
        prompt_via_stdin=True,
    ),
    "codex-gpt-5.4-mini": AgentSpec(
        id="codex-gpt-5.4-mini",
        label="Codex GPT-5.4 Mini",
        provider="codex",
        model_id="gpt-5.4-mini",
        command=[
            "codex",
            "exec",
            "--json",
            "--ignore-user-config",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            "gpt-5.4-mini",
            "-",
        ],
        prompt_via_stdin=True,
    ),
    "claude-opus-4.8-high": AgentSpec(
        id="claude-opus-4.8-high",
        label="Claude Opus 4.8 High",
        provider="claude",
        model_id="claude-opus-4-8",
        command=[
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
        ],
    ),
    "claude-sonnet-4.6-high": AgentSpec(
        id="claude-sonnet-4.6-high",
        label="Claude Sonnet 4.6 High",
        provider="claude",
        model_id="claude-sonnet-4-6",
        command=[
            "claude",
            "-p",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            "claude-sonnet-4-6",
            "--effort",
            "high",
            "--output-format",
            "stream-json",
            "--verbose",
            "{prompt}",
        ],
    ),
    "cursor-composer-2.5": _cursor_agent(
        id="cursor-composer-2.5",
        label="Cursor Composer 2.5",
        model_id="composer-2.5",
    ),
    "cursor-grok-4.3": _cursor_agent(
        id="cursor-grok-4.3",
        label="Cursor Grok 4.3",
        model_id="grok-4.3",
    ),
    "cursor-kimi-k2.5": _cursor_agent(
        id="cursor-kimi-k2.5",
        label="Cursor Kimi K2.5",
        model_id="kimi-k2.5",
    ),
    "cursor-gemini-3.5-flash": _cursor_agent(
        id="cursor-gemini-3.5-flash",
        label="Cursor Gemini 3.5 Flash",
        model_id="gemini-3.5-flash",
    ),
    "openrouter-deepseek-v4-pro": _opencode_agent(
        id="openrouter-deepseek-v4-pro",
        label="OpenRouter DeepSeek V4 Pro",
        model_id="openrouter/deepseek/deepseek-v4-pro",
    ),
    "openrouter-qwen3.7-max": _opencode_agent(
        id="openrouter-qwen3.7-max",
        label="OpenRouter Qwen 3.7 Max",
        model_id="openrouter/qwen/qwen3.7-max",
    ),
}


DEFAULT_AGENT_IDS = list(DEFAULT_AGENTS)


def load_agent_file(path: Path) -> dict[str, AgentSpec]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"Agent file contains invalid YAML: {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Agent file must contain a mapping: {path}")
    agents = raw.get("agents", raw)
    if not isinstance(agents, dict):
        raise ValueError(f"Agent file must contain a mapping: {path}")

    loaded: dict[str, AgentSpec] = {}
    for agent_id, value in agents.items():
        if not isinstance(value, dict):
            raise ValueError(f"Agent {agent_id!r} must be a mapping.")
        data: dict[str, Any] = {"id": agent_id, **value}
        loaded[agent_id] = AgentSpec.model_validate(data)
    return loaded


def available_agents(
    *,
    agent_file: Path | None = None,
    extra_files: list[Path] | None = None,
) -> dict[str, AgentSpec]:
    """Return defaults merged with any user-supplied agent files."""
    available = dict(DEFAULT_AGENTS)
    for path in extra_files or []:
        available.update(load_agent_file(path))
    if agent_file:
        available.update(load_agent_file(agent_file))
    return available


def resolve_agents(
    agent_ids: list[str] | None,
    agent_file: Path | None = None,
    *,
    extra_files: list[Path] | None = None,
    all_agents: bool = False,
    providers: list[str] | None = None,
    exclude: list[str] | None = None,
    default_set: list[str] | None = None,
) -> list[AgentSpec]:
    """Resolve a list of agents from CLI / config inputs.

    Selection priority (first match wins):
    1. Explicit `agent_ids` from `--agent`
    2. `providers` from `--provider`
    3. `all_agents` from `--all-agents`
    4. `default_set` from config `[selection].default_agents`
    5. Full registry (preserves prior behavior)

    `exclude` is applied after selection.
    """
    available = available_agents(agent_file=agent_file, extra_files=extra_files)
    known = ", ".join(sorted(available))

    if agent_ids:
        missing = [agent_id for agent_id in agent_ids if agent_id not in available]
        if missing:
            raise KeyError(f"Unknown agent(s): {', '.join(missing)}. Known agents: {known}")
        selected = list(agent_ids)
    elif providers:
        wanted = {p.strip().lower() for p in providers}
        selected = [
            agent_id for agent_id, spec in available.items() if spec.provider.lower() in wanted
        ]
        if not selected:
            raise KeyError(f"No agents match provider(s) {sorted(wanted)!r}. Known agents: {known}")
    elif all_agents:
        selected = list(available.keys())
    elif default_set:
        missing = [agent_id for agent_id in default_set if agent_id not in available]
        if missing:
            raise KeyError(
                f"Config default_agents references unknown agent(s): {', '.join(missing)}. "
                f"Known agents: {known}"
            )
        selected = list(default_set)
    else:
        selected = list(DEFAULT_AGENT_IDS)

    if exclude:
        drop = set(exclude)
        unknown = drop - set(available.keys())
        if unknown:
            raise KeyError(
                f"--exclude-agent references unknown agent(s): {', '.join(sorted(unknown))}"
            )
        selected = [agent_id for agent_id in selected if agent_id not in drop]
        if not selected:
            raise KeyError("Agent selection is empty after applying --exclude-agent.")

    return [available[agent_id] for agent_id in selected]
