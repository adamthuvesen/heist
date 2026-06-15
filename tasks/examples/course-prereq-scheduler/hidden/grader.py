from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grader_safety import safe_result  # noqa: E402


def check(name: str, passed: bool, message: str = "") -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "message": message}


def has_result_shape(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("schedule"), list)
        and isinstance(value.get("deferred"), list)
        and isinstance(value.get("rejections"), dict)
    )


def safe_run(workspace: Path, courses: list[dict], config: dict) -> object:
    sys.path.insert(0, str(workspace / "src"))
    try:
        try:
            from scheduler import plan_schedule

            result = plan_schedule(courses, config)
            if not has_result_shape(result):
                return safe_result(None)
            return safe_result(result)
        except Exception:
            return safe_result(None)
    finally:
        sys.path.pop(0)
        for module in ("scheduler", "scheduler.planner", "scheduler.graph"):
            sys.modules.pop(module, None)


def main() -> None:
    workspace = Path(sys.argv[1])
    checks: list[dict[str, object]] = []

    # 1. Linear chain: A → B → C
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": []},
            {"id": "B", "credits": 3, "prereqs": ["A"]},
            {"id": "C", "credits": 3, "prereqs": ["B"]},
        ],
        {"max_credits_per_quarter": 12, "max_quarters": 6},
    )
    checks.append(
        check(
            "linear_chain_three_quarters",
            result["schedule"] == [["A"], ["B"], ["C"]]
            and result["deferred"] == []
            and result["rejections"] == {},
            str(result),
        )
    )

    # 2. Parallel ready courses pack into one quarter
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": []},
            {"id": "B", "credits": 3, "prereqs": []},
            {"id": "C", "credits": 3, "prereqs": []},
        ],
        {"max_credits_per_quarter": 9, "max_quarters": 4},
    )
    checks.append(
        check(
            "parallel_ready_courses_pack_into_one_quarter",
            result["schedule"] == [["A", "B", "C"]] and result["rejections"] == {},
            str(result),
        )
    )

    # 3. Credit cap splits ready courses across quarters
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": []},
            {"id": "B", "credits": 3, "prereqs": []},
            {"id": "C", "credits": 3, "prereqs": []},
            {"id": "D", "credits": 3, "prereqs": []},
        ],
        {"max_credits_per_quarter": 6, "max_quarters": 4},
    )
    checks.append(
        check(
            "credit_cap_splits_ready_courses",
            result["schedule"] == [["A", "B"], ["C", "D"]],
            str(result),
        )
    )

    # 4. Prereq blocks same-quarter scheduling
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": []},
            {"id": "B", "credits": 3, "prereqs": ["A"]},
        ],
        {"max_credits_per_quarter": 12, "max_quarters": 4},
    )
    checks.append(
        check(
            "prereq_blocks_same_quarter",
            result["schedule"] == [["A"], ["B"]],
            str(result),
        )
    )

    # 5. Cycle reports error
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": ["B"]},
            {"id": "B", "credits": 3, "prereqs": ["A"]},
        ],
        {"max_credits_per_quarter": 12, "max_quarters": 4},
    )
    checks.append(
        check(
            "cycle_reports_error",
            result["rejections"].get("A") == "cycle"
            and result["rejections"].get("B") == "cycle"
            and result["schedule"] == []
            and result["deferred"] == [],
            str(result),
        )
    )

    # 6. Missing prereq reports error
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": ["X"]},
        ],
        {"max_credits_per_quarter": 12, "max_quarters": 4},
    )
    checks.append(
        check(
            "missing_prereq_reports_error",
            result["rejections"].get("A") == "missing_prereq" and result["schedule"] == [],
            str(result),
        )
    )

    # 7. Overweight course reports error
    result = safe_run(
        workspace,
        [
            {"id": "BIG", "credits": 9, "prereqs": []},
            {"id": "OK", "credits": 3, "prereqs": []},
        ],
        {"max_credits_per_quarter": 6, "max_quarters": 4},
    )
    checks.append(
        check(
            "overweight_course_reports_error",
            result["rejections"].get("BIG") == "overweight" and result["schedule"] == [["OK"]],
            str(result),
        )
    )

    # 8. Deferred when max_quarters exceeded
    result = safe_run(
        workspace,
        [
            {"id": "A", "credits": 3, "prereqs": []},
            {"id": "B", "credits": 3, "prereqs": ["A"]},
            {"id": "C", "credits": 3, "prereqs": ["B"]},
            {"id": "D", "credits": 3, "prereqs": ["C"]},
            {"id": "E", "credits": 3, "prereqs": ["D"]},
        ],
        {"max_credits_per_quarter": 12, "max_quarters": 3},
    )
    checks.append(
        check(
            "deferred_when_max_quarters_exceeded",
            result["schedule"] == [["A"], ["B"], ["C"]]
            and result["deferred"] == ["D", "E"]
            and result["rejections"] == {},
            str(result),
        )
    )

    # 9. Course with errored prereq goes to deferred, not errored itself
    result = safe_run(
        workspace,
        [
            {"id": "BIG", "credits": 9, "prereqs": []},
            {"id": "OK", "credits": 3, "prereqs": ["BIG"]},
        ],
        {"max_credits_per_quarter": 6, "max_quarters": 4},
    )
    checks.append(
        check(
            "course_with_errored_prereq_deferred_not_errored",
            result["rejections"].get("BIG") == "overweight"
            and "OK" not in result["rejections"]
            and "OK" in result["deferred"]
            and result["schedule"] == [],
            str(result),
        )
    )

    # 10. Generated deterministic schedule: 8 courses, cap=6, 3 chains
    courses = [
        {"id": "A", "credits": 3, "prereqs": []},
        {"id": "B", "credits": 3, "prereqs": []},
        {"id": "C", "credits": 3, "prereqs": []},
        {"id": "D", "credits": 3, "prereqs": []},
        {"id": "E", "credits": 3, "prereqs": []},
        {"id": "F", "credits": 3, "prereqs": ["A"]},
        {"id": "G", "credits": 3, "prereqs": ["B"]},
        {"id": "H", "credits": 3, "prereqs": ["C"]},
    ]
    result = safe_run(
        workspace,
        courses,
        {"max_credits_per_quarter": 6, "max_quarters": 10},
    )
    expected_schedule = [["A", "B"], ["C", "D"], ["E", "F"], ["G", "H"]]
    checks.append(
        check(
            "generated_deterministic_schedule",
            result["schedule"] == expected_schedule
            and result["deferred"] == []
            and result["rejections"] == {},
            f"got {result['schedule']}",
        )
    )

    score = sum(item["passed"] for item in checks) / len(checks)
    print(json.dumps({"score": score, "passed": score == 1.0, "checks": checks}))


if __name__ == "__main__":
    main()
