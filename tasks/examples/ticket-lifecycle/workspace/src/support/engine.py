from __future__ import annotations


def process_tickets(
    events: list[dict[str, object]],
    config: dict[str, object],
    now: str,
) -> dict[str, object]:
    tickets: dict[str, dict[str, object]] = {}
    transitions: list[dict[str, object]] = []
    for event in events:
        ticket_id = str(event.get("ticket_id", ""))
        typ = event.get("type")
        if typ == "opened":
            tickets[ticket_id] = {
                "state": "open",
                "priority": event.get("priority"),
                "assignee": event.get("assignee"),
                "resolution": None,
                "opened_at": event.get("at"),
                "resolved_at": None,
            }
        elif typ == "resolved" and ticket_id in tickets:
            previous = tickets[ticket_id]["state"]
            tickets[ticket_id]["state"] = "resolved"
            tickets[ticket_id]["resolution"] = event.get("resolution")
            tickets[ticket_id]["resolved_at"] = event.get("at")
            transitions.append(
                {
                    "ticket": ticket_id,
                    "from": previous,
                    "to": "resolved",
                    "at": event.get("at"),
                    "event": str(event.get("id", "")),
                }
            )
    return {"tickets": tickets, "transitions": transitions, "rejections": {}}
