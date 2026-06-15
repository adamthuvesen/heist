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
    return isinstance(value.get("deliveries"), list) and isinstance(value.get("rejections"), dict)


def safe_run(
    workspace: Path,
    messages: list[dict],
    users: dict,
    channels: dict,
    rules: list,
    now: str,
) -> object:
    sys.path.insert(0, str(workspace / "src"))
    try:
        try:
            from notify import route_notifications

            result = route_notifications(messages, users, channels, rules, now)
            if not has_result_shape(result):
                return safe_result(None)
            return safe_result(result)
        except Exception:
            return safe_result(None)
    finally:
        sys.path.pop(0)
        for module in ("notify", "notify.router"):
            sys.modules.pop(module, None)


def main() -> None:
    workspace = Path(sys.argv[1])
    checks: list[dict[str, object]] = []

    email_chan = {"email": {"capabilities": ["marketing", "transactional", "digest"]}}
    sms_chan = {"sms": {"capabilities": ["alert", "transactional"]}}
    push_chan = {"push": {"capabilities": ["alert", "marketing", "transactional"]}}
    all_channels = {**email_chan, **sms_chan, **push_chan}

    # 1. Simple match → primary channel
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "marketing",
                "priority": "normal",
                "sent_at": "2026-05-01T10:00:00Z",
            }
        ],
        {"u1": {"timezone": "+00:00", "channels": {"email": {"enabled": True}}}},
        email_chan,
        [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}],
        "2026-05-01T10:00:00Z",
    )
    checks.append(
        check(
            "simple_match_routes_to_primary_channel",
            result["deliveries"] == [{"message": "m1", "channel": "email", "user": "u1"}]
            and result["rejections"] == {},
            str(result),
        )
    )

    # 2. First matching rule wins (seed picks last)
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "alert",
                "priority": "normal",
                "sent_at": "2026-05-01T10:00:00Z",
            }
        ],
        {
            "u1": {
                "timezone": "+00:00",
                "channels": {"push": {"enabled": True}, "sms": {"enabled": True}},
            }
        },
        {**push_chan, **sms_chan},
        [
            {"match": {"type": "alert"}, "route": ["push"], "fallback": []},
            {"match": {"type": "alert"}, "route": ["sms"], "fallback": []},
        ],
        "2026-05-01T10:00:00Z",
    )
    checks.append(
        check(
            "first_matching_rule_wins",
            result["deliveries"] == [{"message": "m1", "channel": "push", "user": "u1"}],
            str(result),
        )
    )

    # 3. Capability filter skips incapable channel → uses fallback
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "marketing",
                "priority": "normal",
                "sent_at": "2026-05-01T10:00:00Z",
            }
        ],
        {
            "u1": {
                "timezone": "+00:00",
                "channels": {"sms": {"enabled": True}, "email": {"enabled": True}},
            }
        },
        {**sms_chan, **email_chan},
        # primary sms doesn't carry marketing → fallback email
        [{"match": {"type": "marketing"}, "route": ["sms"], "fallback": ["email"]}],
        "2026-05-01T10:00:00Z",
    )
    checks.append(
        check(
            "capability_filter_falls_back",
            result["deliveries"] == [{"message": "m1", "channel": "email", "user": "u1"}],
            str(result),
        )
    )

    # 4. DND suppresses non-urgent message
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "marketing",
                "priority": "normal",
                "sent_at": "2026-05-01T22:30:00Z",
            }
        ],
        {
            "u1": {
                "timezone": "+00:00",
                "channels": {"email": {"enabled": True}},
                "dnd": {"start": "22:00", "end": "23:00"},
            }
        },
        email_chan,
        [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}],
        "2026-05-01T22:30:00Z",
    )
    checks.append(
        check(
            "dnd_suppresses_non_urgent",
            result["deliveries"] == [] and result["rejections"].get("m1") == "dnd",
            str(result),
        )
    )

    # 5. Urgent overrides DND
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "alert",
                "priority": "urgent",
                "sent_at": "2026-05-01T22:30:00Z",
            }
        ],
        {
            "u1": {
                "timezone": "+00:00",
                "channels": {"push": {"enabled": True}},
                "dnd": {"start": "22:00", "end": "23:00"},
            }
        },
        push_chan,
        [{"match": {"type": "alert"}, "route": ["push"], "fallback": []}],
        "2026-05-01T22:30:00Z",
    )
    checks.append(
        check(
            "urgent_overrides_dnd",
            result["deliveries"] == [{"message": "m1", "channel": "push", "user": "u1"}]
            and result["rejections"] == {},
            str(result),
        )
    )

    # 6. Overnight DND window wraps correctly (22:00-07:00, message at 03:00 local)
    result = safe_run(
        workspace,
        [
            {
                "id": "m1",
                "recipient": "u1",
                "type": "marketing",
                "priority": "normal",
                "sent_at": "2026-05-01T03:00:00Z",
            }
        ],
        {
            "u1": {
                "timezone": "+00:00",
                "channels": {"email": {"enabled": True}},
                "dnd": {"start": "22:00", "end": "07:00"},
            }
        },
        email_chan,
        [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}],
        "2026-05-01T03:00:00Z",
    )
    checks.append(
        check(
            "overnight_dnd_window_wraps",
            result["deliveries"] == [] and result["rejections"].get("m1") == "dnd",
            str(result),
        )
    )

    # 7. Overnight DND boundaries — start inclusive, end exclusive
    boundary_msg = {
        "type": "marketing",
        "priority": "normal",
        "recipient": "u1",
    }
    user_overnight = {
        "u1": {
            "timezone": "+00:00",
            "channels": {"email": {"enabled": True}},
            "dnd": {"start": "22:00", "end": "07:00"},
        }
    }
    boundary_rules = [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}]
    at_start = safe_run(
        workspace,
        [{**boundary_msg, "id": "ms", "sent_at": "2026-05-01T22:00:00Z"}],
        user_overnight,
        email_chan,
        boundary_rules,
        "2026-05-01T22:00:00Z",
    )
    at_end = safe_run(
        workspace,
        [{**boundary_msg, "id": "me", "sent_at": "2026-05-01T07:00:00Z"}],
        user_overnight,
        email_chan,
        boundary_rules,
        "2026-05-01T07:00:00Z",
    )
    checks.append(
        check(
            "overnight_dnd_boundaries_inclusive_start_exclusive_end",
            at_start["rejections"].get("ms") == "dnd"
            and len(at_start["deliveries"]) == 0
            and at_end["rejections"] == {}
            and len(at_end["deliveries"]) == 1,
            f"start={at_start} end={at_end}",
        )
    )

    # 8. Duplicate message id suppressed
    duplicate_msg = {
        "id": "m1",
        "recipient": "u1",
        "type": "marketing",
        "priority": "normal",
        "sent_at": "2026-05-01T10:00:00Z",
    }
    result = safe_run(
        workspace,
        [duplicate_msg, dict(duplicate_msg)],
        {"u1": {"timezone": "+00:00", "channels": {"email": {"enabled": True}}}},
        email_chan,
        [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}],
        "2026-05-01T10:00:00Z",
    )
    checks.append(
        check(
            "duplicate_message_id_suppressed",
            len(result["deliveries"]) == 1 and result["rejections"].get("m1") == "duplicate",
            str(result),
        )
    )

    # 9. Generated deterministic routing: 12 messages across 4 users
    users = {}
    for i in range(4):
        users[f"u{i}"] = {
            "timezone": "+00:00",
            "channels": {"email": {"enabled": True}, "push": {"enabled": True}},
        }
    messages = []
    for i in range(12):
        messages.append(
            {
                "id": f"msg-{i:02d}",
                "recipient": f"u{i % 4}",
                "type": "marketing" if i % 2 == 0 else "alert",
                "priority": "normal",
                "sent_at": f"2026-05-01T{10 + (i % 8):02d}:00:00Z",
            }
        )
    # Shuffle deterministically by reversing
    shuffled = list(reversed(messages))
    rules = [
        {"match": {"type": "marketing"}, "route": ["email"], "fallback": []},
        {"match": {"type": "alert"}, "route": ["push"], "fallback": []},
    ]
    result = safe_run(workspace, shuffled, users, all_channels, rules, "2026-05-01T20:00:00Z")
    delivered_ids = [d["message"] for d in result["deliveries"]]
    expected_ids = sorted(
        [m["id"] for m in messages],
        key=lambda mid: (next(m["sent_at"] for m in messages if m["id"] == mid), mid),
    )
    checks.append(
        check(
            "generated_deterministic_sort_order",
            delivered_ids == expected_ids
            and len(result["deliveries"]) == 12
            and result["rejections"] == {},
            f"got {delivered_ids[:4]}... expected {expected_ids[:4]}...",
        )
    )

    score = sum(item["passed"] for item in checks) / len(checks)
    print(json.dumps({"score": score, "passed": score == 1.0, "checks": checks}))


if __name__ == "__main__":
    main()
