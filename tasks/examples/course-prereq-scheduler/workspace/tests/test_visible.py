from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scheduler import plan_schedule


def test_linear_chain_schedules_one_course_per_quarter() -> None:
    courses = [
        {"id": "A", "credits": 3, "prereqs": []},
        {"id": "B", "credits": 3, "prereqs": ["A"]},
        {"id": "C", "credits": 3, "prereqs": ["B"]},
    ]
    config = {"max_credits_per_quarter": 12, "max_quarters": 6}

    result = plan_schedule(courses, config)

    assert result["schedule"] == [["A"], ["B"], ["C"]]
    assert result["deferred"] == []
    assert result["rejections"] == {}
