from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

CONFIG_FILENAME = "heist.toml"
PYPROJECT_FILENAME = "pyproject.toml"
ENV_PREFIX = "HEIST_"


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str = "smoke"
    jobs: PositiveInt = 1
    timeout_s: PositiveInt = 1800
    output_dir: str = "runs"
    progress: bool = True
    exit_on_failure: bool = False
    # Wrap the agent CLI in a macOS sandbox-exec profile that denies reads of the
    # tasks/ tree (graders + references). macOS only; off by default.
    sandbox: bool = False


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_agents: list[str] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(default_factory=list)


class HeistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    providers: dict[str, PositiveInt] = Field(default_factory=dict)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)


@dataclass
class LoadedConfig:
    """Result of layering config sources, with provenance for `config path`."""

    config: HeistConfig
    sources: list[Path] = field(default_factory=list)

    @property
    def defaults(self) -> DefaultsConfig:
        return self.config.defaults

    @property
    def providers(self) -> dict[str, int]:
        return self.config.providers

    @property
    def selection(self) -> SelectionConfig:
        return self.config.selection

    @property
    def agents(self) -> AgentsConfig:
        return self.config.agents


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def _read_heist_toml(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONFIG_FILENAME
    if not path.exists():
        return {}
    return _read_toml(path)


def _read_pyproject_table(repo_root: Path) -> dict[str, Any]:
    path = repo_root / PYPROJECT_FILENAME
    if not path.exists():
        return {}
    raw = _read_toml(path)
    tool = raw.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    section = tool.get("heist", {})
    return section if isinstance(section, dict) else {}


_KNOWN_DEFAULT_KEYS = set(DefaultsConfig.model_fields)
# Internal env vars HEIST sets per-job (not user-facing config knobs);
# silently ignore them in the overlay so the strict-key check stays useful.
_INTERNAL_ENV_KEYS = {"task_id", "agent_id", "usage_file"}


def _env_overlay() -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    providers: dict[str, int] = {}
    for raw_key, value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        key = raw_key[len(ENV_PREFIX) :].lower()
        if key in _INTERNAL_ENV_KEYS:
            continue
        if key.startswith("provider_"):
            providers[key[len("provider_") :]] = _coerce_positive_int(raw_key, value)
            continue
        if key not in _KNOWN_DEFAULT_KEYS:
            raise ValueError(
                f"unknown {ENV_PREFIX}* env var: {raw_key}. "
                f"Known keys: {sorted(_KNOWN_DEFAULT_KEYS)}."
            )
        defaults[key] = _coerce_value(key, value)
    if defaults:
        overlay["defaults"] = defaults
    if providers:
        overlay["providers"] = providers
    return overlay


def _coerce_value(key: str, value: str) -> Any:
    if key in {"jobs", "timeout_s"}:
        return _coerce_positive_int(key, value)
    if key in {"progress", "exit_on_failure", "sandbox"}:
        return _coerce_bool(key, value)
    return value


def _coerce_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"expected int, got {value!r}") from error


def _coerce_positive_int(key: str, value: str) -> int:
    parsed = _coerce_int(value)
    if parsed < 1:
        raise ValueError(f"{key} must be >= 1, got {parsed}")
    return parsed


def _coerce_bool(key: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{key} must be a boolean: expected one of true/false, yes/no, on/off, 1/0; got {value!r}"
    )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(repo_root: Path) -> LoadedConfig:
    """Layer config sources, low → high priority.

    Priority order: builtin defaults (from `DefaultsConfig` field defaults)
    < [tool.heist] in pyproject.toml < heist.toml at repo root < HEIST_* env
    vars. CLI flags are layered on top by the caller.
    """
    merged: dict[str, Any] = {}
    sources: list[Path] = []

    pyproject_section = _read_pyproject_table(repo_root)
    if pyproject_section:
        merged = _deep_merge(merged, pyproject_section)
        sources.append(repo_root / PYPROJECT_FILENAME)

    heist_section = _read_heist_toml(repo_root)
    if heist_section:
        merged = _deep_merge(merged, heist_section)
        sources.append(repo_root / CONFIG_FILENAME)

    env_section = _env_overlay()
    if env_section:
        merged = _deep_merge(merged, env_section)

    config = HeistConfig.model_validate(merged)
    return LoadedConfig(config=config, sources=sources)


def render_starter_config() -> str:
    """Return the contents of a starter heist.toml."""
    return """# HEIST repo-level defaults.
# CLI flags override these; HEIST_* env vars sit between flags and this file.

[defaults]
suite           = "smoke"
jobs            = 8           # max concurrent (agent, task) jobs across all providers
timeout_s       = 1800        # per-task timeout when task.yaml does not override
output_dir      = "runs"      # where run directories are written
progress        = true        # live rich table when stdout is a TTY
exit_on_failure = false       # set true in CI to fail the process on errors

[providers]                   # per-provider concurrency caps
claude = 3
codex  = 2
cursor = 4
opencode = 2

[selection]
# Agent ids used when `heist run` has no --agent, --provider, or --all-agents.
# Leave empty to use every built-in default agent.
default_agents = [
  "claude-opus-4.8-xhigh",
  "codex-gpt-5.5-high",
  "cursor-composer-2.5",
]

[agents]
# Extra YAML agent files to merge into the registry. Same shape as --agent-file.
files = []
"""


def parse_provider_jobs(spec: str | None) -> dict[str, int]:
    """Parse a `name=N,name=N` string into a dict of per-provider caps."""
    if not spec:
        return {}
    out: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--provider-jobs entry must be name=N, got {item!r}")
        name, _, value = item.partition("=")
        name = name.strip().lower()
        out[name] = _coerce_positive_int(f"--provider-jobs {name!r}", value)
    return out
