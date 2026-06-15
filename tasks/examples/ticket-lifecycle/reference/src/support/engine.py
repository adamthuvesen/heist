from __future__ import annotations

from datetime import UTC, datetime


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def process_tickets(
    events: list[dict[str, object]],
    config: dict[str, object],
    now: str,
) -> dict[str, object]:
    reopen_window = float(config.get("reopen_window_seconds", 0) or 0)
    seen: set[str] = set()
    tickets: dict[str, dict[str, object]] = {}
    transitions: list[dict[str, object]] = []
    rejections: dict[str, str] = {}

    ordered = sorted(
        events,
        key=lambda item: (str(item.get("at", "")), str(item.get("id", ""))),
    )

    for event in ordered:
        event_id = str(event.get("id", ""))
        ticket_id = str(event.get("ticket_id", ""))
        typ = event.get("type")
        at = event.get("at")

        if event_id in seen:
            rejections[event_id] = "duplicate"
            continue
        seen.add(event_id)

        ticket = tickets.get(ticket_id)

        if typ == "opened":
            if ticket is not None:
                rejections[event_id] = "duplicate_open"
                continue
            tickets[ticket_id] = {
                "state": "open",
                "priority": event.get("priority"),
                "assignee": event.get("assignee"),
                "resolution": None,
                "opened_at": at,
                "resolved_at": None,
            }
            continue

        if ticket is None:
            rejections[event_id] = "unknown_ticket"
            continue

        if ticket["state"] == "resolved" and typ != "reopened":
            rejections[event_id] = "terminal"
            continue

        if typ == "assigned":
            previous = ticket["state"]
            ticket["assignee"] = event.get("assignee")
            ticket["state"] = "assigned"
            transitions.append(
                {
                    "ticket": ticket_id,
                    "from": previous,
                    "to": "assigned",
                    "at": at,
                    "event": event_id,
                }
            )
        elif typ == "responded":
            previous = ticket["state"]
            ticket["state"] = "in_progress"
            transitions.append(
                {
                    "ticket": ticket_id,
                    "from": previous,
                    "to": "in_progress",
                    "at": at,
                    "event": event_id,
                }
            )
        elif typ == "resolved":
            previous = ticket["state"]
            ticket["state"] = "resolved"
            ticket["resolution"] = event.get("resolution")
            ticket["resolved_at"] = at
            transitions.append(
                {
                    "ticket": ticket_id,
                    "from": previous,
                    "to": "resolved",
                    "at": at,
                    "event": event_id,
                }
            )
        elif typ == "reopened":
            resolved_dt = _parse_time(ticket.get("resolved_at"))
            reopened_dt = _parse_time(at)
            if (
                resolved_dt is None
                or reopened_dt is None
                or (reopened_dt - resolved_dt).total_seconds() > reopen_window
            ):
                rejections[event_id] = "reopen_window"
                continue
            previous = ticket["state"]
            ticket["state"] = "open"
            ticket["resolution"] = None
            ticket["resolved_at"] = None
            transitions.append(
                {"ticket": ticket_id, "from": previous, "to": "open", "at": at, "event": event_id}
            )
        else:
            rejections[event_id] = "unknown_ticket"

    return {"tickets": tickets, "transitions": transitions, "rejections": rejections}
