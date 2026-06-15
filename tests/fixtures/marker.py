from __future__ import annotations

import sys
from pathlib import Path

from heist.models import AgentSpec

FAKE_AGENT_SCRIPT = Path(__file__).parent / "fake_agent.py"


def write_marker_task(root: Path, task_id: str = "marker") -> None:
    task_dir = root / "tasks" / "smoke" / task_id
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "hidden").mkdir()
    (task_dir / "reference").mkdir()
    (task_dir / "workspace" / "answer.txt").write_text("no\n")
    (task_dir / "reference" / "answer.txt").write_text("yes\n")
    (task_dir / "task.yaml").write_text(
        f"""
id: {task_id}
title: Marker task
category: fake
prompt: Write yes to answer.txt.
visible_test_command: ["python", "-c", "print('visible ok')"]
"""
    )
    (task_dir / "hidden" / "grader.py").write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
passed = (workspace / "answer.txt").read_text().strip() == "yes"
print(json.dumps({
    "score": 1.0 if passed else 0.0,
    "passed": passed,
    "checks": [{"name": "answer", "passed": passed, "message": ""}],
}))
"""
    )


def fake_agent(mode: str, model_id: str = "fake-model") -> AgentSpec:
    return AgentSpec(
        id=f"fake-{mode}",
        label=f"Fake {mode}",
        provider="fake",
        model_id=model_id,
        command=[sys.executable, str(FAKE_AGENT_SCRIPT), mode],
    )


def break_grader(root: Path, task_id: str = "marker") -> None:
    grader = root / "tasks" / "smoke" / task_id / "hidden" / "grader.py"
    grader.write_text(
        """
from __future__ import annotations

raise RuntimeError("grader exploded")
"""
    )
