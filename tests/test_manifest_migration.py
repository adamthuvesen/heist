from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from heist.models import RUN_MANIFEST_SCHEMA_VERSION, RunManifest
from heist.runner import load_manifest


def _v1_payload(run_id: str = "abc") -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite": "smoke",
        "agent_ids": ["fake-pass"],
        "task_ids": ["marker"],
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "duration_s": None,
        "repo_root": "/tmp/root",
        "run_dir": "/tmp/run",
        "default_agents": [],
        "status": "completed",
    }


def _write_manifest(dir_: Path, payload: dict[str, object]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_v2_round_trip_does_not_rewrite_file(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="r",
        suite="smoke",
        agent_ids=[],
        task_ids=[],
        repo_root=str(tmp_path),
        run_dir=str(tmp_path),
        default_agents=[],
    )
    assert manifest.schema_version == RUN_MANIFEST_SCHEMA_VERSION == 2
    assert manifest.harness_git_sha is None
    assert manifest.tags == []
    assert manifest.source_run_id is None
    assert manifest.kind == "live"

    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    before = path.read_bytes()

    loaded = load_manifest(tmp_path)
    assert loaded.run_id == "r"
    # Idempotent: a manifest already at the current version must not be rewritten.
    assert path.read_bytes() == before


def test_v1_manifest_is_migrated_in_place(tmp_path: Path) -> None:
    _write_manifest(tmp_path, _v1_payload())

    loaded = load_manifest(tmp_path)
    assert loaded.schema_version == RUN_MANIFEST_SCHEMA_VERSION
    assert loaded.harness_git_sha is None
    assert loaded.tags == []
    assert loaded.source_run_id is None
    assert loaded.kind == "live"

    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["schema_version"] == 2
    for key in ("harness_git_sha", "tags", "source_run_id", "kind"):
        assert key in persisted


def test_v1_migration_is_idempotent_on_reload(tmp_path: Path) -> None:
    _write_manifest(tmp_path, _v1_payload())

    load_manifest(tmp_path)
    snapshot = (tmp_path / "manifest.json").read_bytes()

    load_manifest(tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == snapshot


def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    payload = _v1_payload()
    payload["schema_version"] = RUN_MANIFEST_SCHEMA_VERSION + 1
    _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match=r"schema_version=\d+ is incompatible"):
        load_manifest(tmp_path)


def test_non_object_manifest_payload_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="not a JSON object"):
        load_manifest(tmp_path)


def test_replay_manifest_round_trip(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="replay-1",
        suite="smoke",
        agent_ids=["fake-pass"],
        task_ids=["marker"],
        repo_root=str(tmp_path),
        run_dir=str(tmp_path),
        default_agents=[],
        kind="replay",
        source_run_id="live-source",
    )
    assert manifest.kind == "replay"
    assert manifest.source_run_id == "live-source"

    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    loaded = load_manifest(tmp_path)
    assert loaded.kind == "replay"
    assert loaded.source_run_id == "live-source"


def test_v1_migrated_manifest_defaults_to_kind_live(tmp_path: Path) -> None:
    _write_manifest(tmp_path, _v1_payload())
    loaded = load_manifest(tmp_path)
    assert loaded.kind == "live"
    assert loaded.source_run_id is None
