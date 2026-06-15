from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support import process_tickets


def test_open_then_resolve_marks_ticket_resolved() -> None:
    events = [
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
    ]
    result = process_tickets(events, {"reopen_window_seconds": 60}, "2026-05-03T00:00:00Z")
    assert result["tickets"]["t1"]["state"] == "resolved"
    assert result["tickets"]["t1"]["resolution"] == "fixed"
    assert result["tickets"]["t1"]["resolved_at"] == "2026-05-02T00:00:00Z"
    assert result["transitions"][0]["to"] == "resolved"
    assert result["rejections"] == {}
