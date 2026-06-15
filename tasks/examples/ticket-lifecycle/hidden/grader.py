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
        isinstance(value.get("tickets"), dict)
        and isinstance(value.get("transitions"), list)
        and isinstance(value.get("rejections"), dict)
    )


def safe_run(workspace: Path, events: list[dict], config: dict, now: str) -> object:
    sys.path.insert(0, str(workspace / "src"))
    try:
        try:
            from support import process_tickets

            result = process_tickets(events, config, now)
            if not has_result_shape(result):
                return safe_result(None)
            return safe_result(result)
        except Exception:
            return safe_result(None)
    finally:
        sys.path.pop(0)
        for module in ("support", "support.engine"):
            sys.modules.pop(module, None)


def main() -> None:
    workspace = Path(sys.argv[1])
    checks: list[dict[str, object]] = []
    config = {"reopen_window_seconds": 60}

    # 1. Happy path: single open + resolve
    result = safe_run(
        workspace,
        [
            {
                "id": "e1",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p2",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "e2",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fixed",
                "at": "2026-05-02T00:00:00Z",
            },
        ],
        config,
        "2026-05-03T00:00:00Z",
    )
    checks.append(
        check(
            "happy_path_open_then_resolved",
            result["tickets"]["t1"]["state"] == "resolved"
            and result["tickets"]["t1"]["resolution"] == "fixed"
            and result["tickets"]["t1"]["resolved_at"] == "2026-05-02T00:00:00Z"
            and result["rejections"] == {},
            str(result),
        )
    )

    # 2. Full lifecycle: open → assigned → responded → resolved
    result = safe_run(
        workspace,
        [
            {
                "id": "e1",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p1",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "e2",
                "ticket_id": "t1",
                "type": "assigned",
                "assignee": "alice",
                "at": "2026-05-01T01:00:00Z",
            },
            {
                "id": "e3",
                "ticket_id": "t1",
                "type": "responded",
                "author": "alice",
                "at": "2026-05-01T02:00:00Z",
            },
            {
                "id": "e4",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "answered",
                "at": "2026-05-01T03:00:00Z",
            },
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    transitions = result["transitions"]
    checks.append(
        check(
            "full_lifecycle_records_three_transitions",
            len(transitions) == 3
            and [t["to"] for t in transitions] == ["assigned", "in_progress", "resolved"]
            and result["tickets"]["t1"]["assignee"] == "alice"
            and result["tickets"]["t1"]["state"] == "resolved",
            str(result),
        )
    )

    # 3. Chronological replay: input in reverse chronological order
    result = safe_run(
        workspace,
        [
            {
                "id": "later",
                "ticket_id": "t2",
                "type": "opened",
                "priority": "p2",
                "at": "2026-05-02T00:00:00Z",
            },
            {
                "id": "earlier",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p1",
                "at": "2026-05-01T00:00:00Z",
            },
        ],
        config,
        "2026-05-03T00:00:00Z",
    )
    checks.append(
        check(
            "chronological_replay_keys_in_at_order",
            list(result["tickets"].keys()) == ["t1", "t2"],
            str(result),
        )
    )

    # 4. Duplicate event id is ignored (second event with same id is rejected)
    result = safe_run(
        workspace,
        [
            {
                "id": "dup",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p3",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "dup",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fast",
                "at": "2026-05-01T01:00:00Z",
            },
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "duplicate_event_id_ignored",
            result["tickets"]["t1"]["state"] == "open"
            and result["tickets"]["t1"]["resolution"] is None
            and result["rejections"].get("dup") == "duplicate"
            and result["transitions"] == [],
            str(result),
        )
    )

    # 5. Unknown ticket rejection
    result = safe_run(
        workspace,
        [
            {
                "id": "ghost",
                "ticket_id": "missing",
                "type": "resolved",
                "resolution": "n/a",
                "at": "2026-05-01T00:00:00Z",
            },
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "unknown_ticket_rejected",
            result["rejections"].get("ghost") == "unknown_ticket" and result["tickets"] == {},
            str(result),
        )
    )

    # 6. Terminal state rejection: assigned event after resolved
    result = safe_run(
        workspace,
        [
            {
                "id": "o",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p2",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "r",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fixed",
                "at": "2026-05-01T01:00:00Z",
            },
            {
                "id": "a",
                "ticket_id": "t1",
                "type": "assigned",
                "assignee": "bob",
                "at": "2026-05-01T02:00:00Z",
            },
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "terminal_state_rejects_later_event",
            result["rejections"].get("a") == "terminal"
            and result["tickets"]["t1"]["state"] == "resolved",
            str(result),
        )
    )

    # 7. Reopen within window accepted; resolution cleared; opened_at preserved
    result = safe_run(
        workspace,
        [
            {
                "id": "o",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p1",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "r",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fixed",
                "at": "2026-05-01T00:00:30Z",
            },
            {"id": "rop", "ticket_id": "t1", "type": "reopened", "at": "2026-05-01T00:01:00Z"},
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "reopen_within_window_accepted",
            result["tickets"]["t1"]["state"] == "open"
            and result["tickets"]["t1"]["resolution"] is None
            and result["tickets"]["t1"]["resolved_at"] is None
            and result["tickets"]["t1"]["opened_at"] == "2026-05-01T00:00:00Z"
            and "rop" not in result["rejections"]
            and result["transitions"][-1]["to"] == "open",
            str(result),
        )
    )

    # 8. Reopen at exact window boundary accepted (delta == reopen_window_seconds)
    result = safe_run(
        workspace,
        [
            {
                "id": "o",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p1",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "r",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fixed",
                "at": "2026-05-01T00:00:30Z",
            },
            {
                "id": "rop",
                "ticket_id": "t1",
                "type": "reopened",
                "at": "2026-05-01T00:01:30Z",
            },  # exactly 60s after resolved
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "reopen_at_exact_window_boundary_accepted",
            result["tickets"]["t1"]["state"] == "open" and "rop" not in result["rejections"],
            str(result),
        )
    )

    # 9. Reopen outside window rejected; state stays resolved
    result = safe_run(
        workspace,
        [
            {
                "id": "o",
                "ticket_id": "t1",
                "type": "opened",
                "priority": "p1",
                "at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "r",
                "ticket_id": "t1",
                "type": "resolved",
                "resolution": "fixed",
                "at": "2026-05-01T00:00:30Z",
            },
            {"id": "rop", "ticket_id": "t1", "type": "reopened", "at": "2026-05-01T00:02:30Z"},
        ],
        config,
        "2026-05-02T00:00:00Z",
    )
    checks.append(
        check(
            "reopen_outside_window_rejected",
            result["rejections"].get("rop") == "reopen_window"
            and result["tickets"]["t1"]["state"] == "resolved"
            and result["tickets"]["t1"]["resolution"] == "fixed",
            str(result),
        )
    )

    # 10. Generated case: 10 tickets × 4 events, shuffled. Verify ordering and
    # transition count.
    raw: list[dict[str, object]] = []
    for i in range(10):
        ticket = f"t{i:02d}"
        base_hour = i  # one ticket opens per hour so ordering is well-defined
        raw.append(
            {
                "id": f"{ticket}-open",
                "ticket_id": ticket,
                "type": "opened",
                "priority": "p2",
                "at": f"2026-05-01T{base_hour:02d}:00:00Z",
            }
        )
        raw.append(
            {
                "id": f"{ticket}-assign",
                "ticket_id": ticket,
                "type": "assigned",
                "assignee": f"agent-{i % 3}",
                "at": f"2026-05-01T{base_hour:02d}:15:00Z",
            }
        )
        raw.append(
            {
                "id": f"{ticket}-respond",
                "ticket_id": ticket,
                "type": "responded",
                "author": f"agent-{i % 3}",
                "at": f"2026-05-01T{base_hour:02d}:30:00Z",
            }
        )
        raw.append(
            {
                "id": f"{ticket}-resolve",
                "ticket_id": ticket,
                "type": "resolved",
                "resolution": "done",
                "at": f"2026-05-01T{base_hour:02d}:45:00Z",
            }
        )
    # deterministic shuffle: reverse the list
    shuffled = list(reversed(raw))
    result = safe_run(workspace, shuffled, config, "2026-05-02T00:00:00Z")
    expected_keys = [f"t{i:02d}" for i in range(10)]
    checks.append(
        check(
            "generated_shuffled_replay_deterministic",
            list(result["tickets"].keys()) == expected_keys
            and len(result["transitions"]) == 30
            and result["rejections"] == {}
            and all(t["state"] == "resolved" for t in result["tickets"].values()),
            f"keys={list(result['tickets'].keys())[:5]} transitions={len(result['transitions'])}",
        )
    )

    score = sum(item["passed"] for item in checks) / len(checks)
    print(json.dumps({"score": score, "passed": score == 1.0, "checks": checks}))


if __name__ == "__main__":
    main()
