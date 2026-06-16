from __future__ import annotations

from pathlib import Path

import pytest

from heist.config import (
    DefaultsConfig,
    load_config,
    parse_provider_jobs,
    render_starter_config,
)


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def test_load_config_falls_through_to_builtin_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path).config
    expected = DefaultsConfig()
    assert cfg.defaults.suite == expected.suite
    assert cfg.defaults.jobs == expected.jobs
    assert cfg.defaults.timeout_s == 1800
    assert cfg.providers == {}


def test_pyproject_table_layered_below_heist_toml(tmp_path: Path) -> None:
    _write(
        tmp_path / "pyproject.toml",
        "[tool.heist.defaults]\n"
        'suite = "frontier"\n'
        "jobs = 4\n"
        "\n"
        "[tool.heist.providers]\n"
        "claude = 2\n",
    )
    _write(
        tmp_path / "heist.toml",
        "[defaults]\njobs = 12\n\n[providers]\ncursor = 5\n",
    )
    cfg = load_config(tmp_path).config
    assert cfg.defaults.suite == "frontier"
    assert cfg.defaults.jobs == 12
    assert cfg.providers == {"claude": 2, "cursor": 5}


def test_env_vars_override_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(
        tmp_path / "heist.toml",
        '[defaults]\nsuite = "smoke"\njobs = 4\n',
    )
    monkeypatch.setenv("HEIST_SUITE", "frontier")
    monkeypatch.setenv("HEIST_JOBS", "16")
    monkeypatch.setenv("HEIST_PROVIDER_CLAUDE", "7")

    cfg = load_config(tmp_path).config
    assert cfg.defaults.suite == "frontier"
    assert cfg.defaults.jobs == 16
    assert cfg.providers["claude"] == 7


def test_sandbox_default_off_and_env_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_config(tmp_path).config.defaults.sandbox is False
    monkeypatch.setenv("HEIST_SANDBOX", "true")
    assert load_config(tmp_path).config.defaults.sandbox is True


def test_config_rejects_non_positive_file_values(tmp_path: Path) -> None:
    _write(
        tmp_path / "heist.toml",
        "[defaults]\njobs = 0\n\n[providers]\nclaude = -1\n",
    )

    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_load_config_records_source_files(tmp_path: Path) -> None:
    _write(tmp_path / "heist.toml", '[defaults]\nsuite = "frontier"\n')
    loaded = load_config(tmp_path)
    assert tmp_path / "heist.toml" in loaded.sources


def test_parse_provider_jobs_handles_comma_separated() -> None:
    assert parse_provider_jobs("claude=3,cursor=4,codex=2") == {
        "claude": 3,
        "cursor": 4,
        "codex": 2,
    }
    assert parse_provider_jobs(None) == {}
    assert parse_provider_jobs("") == {}


def test_parse_provider_jobs_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError, match="name=N"):
        parse_provider_jobs("claude")
    with pytest.raises(ValueError, match="expected int"):
        parse_provider_jobs("claude=high")
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_provider_jobs("claude=0")


def test_render_starter_config_round_trips_through_load(tmp_path: Path) -> None:
    starter = render_starter_config()
    (tmp_path / "heist.toml").write_text(starter)
    cfg = load_config(tmp_path).config
    assert cfg.defaults.jobs == 8
    assert cfg.defaults.timeout_s == 1800
    assert cfg.providers == {"claude": 3, "codex": 2, "cursor": 4, "opencode": 2}
    assert "claude-opus-4.8-xhigh" in cfg.selection.default_agents
    assert "What `--all-agents` resolves to" not in starter
