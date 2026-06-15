from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the HEIST checkout root from cwd or this package."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "tasks").exists():
            return candidate

    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "tasks").exists():
        return package_root

    raise FileNotFoundError("Could not find HEIST repo root with a tasks/ directory.")


def default_runs_dir(repo_root: Path | None = None) -> Path:
    return (repo_root or find_repo_root()) / "runs"
