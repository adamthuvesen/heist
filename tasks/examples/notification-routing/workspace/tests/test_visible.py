from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from notify import route_notifications


def test_marketing_message_routes_to_enabled_channel() -> None:
    messages = [
        {
            "id": "m1",
            "recipient": "u1",
            "type": "marketing",
            "priority": "normal",
            "sent_at": "2026-05-01T10:00:00Z",
        }
    ]
    users = {
        "u1": {
            "timezone": "+00:00",
            "channels": {"email": {"enabled": True}},
        }
    }
    channels = {"email": {"capabilities": ["marketing", "transactional"]}}
    rules = [{"match": {"type": "marketing"}, "route": ["email"], "fallback": []}]

    result = route_notifications(messages, users, channels, rules, "2026-05-01T10:00:00Z")

    assert result["deliveries"] == [{"message": "m1", "channel": "email", "user": "u1"}]
    assert result["rejections"] == {}
