from __future__ import annotations


def route_notifications(
    messages: list[dict[str, object]],
    users: dict[str, dict[str, object]],
    channels: dict[str, dict[str, object]],
    rules: list[dict[str, object]],
    now: str,
) -> dict[str, object]:
    deliveries: list[dict[str, object]] = []
    rejections: dict[str, str] = {}
    for message in messages:
        msg_id = str(message.get("id", ""))
        recipient = str(message.get("recipient", ""))
        user = users.get(recipient)
        if user is None:
            rejections[msg_id] = "user_unknown"
            continue
        matched: dict[str, object] | None = None
        for rule in rules:
            match = rule.get("match", {})
            if isinstance(match, dict):
                if "type" in match and match["type"] != message.get("type"):
                    continue
                if "priority" in match and match["priority"] != message.get("priority"):
                    continue
            matched = rule
        if matched is None:
            rejections[msg_id] = "no_rule"
            continue
        route = matched.get("route", []) or []
        user_channels = user.get("channels", {}) or {}
        for channel_id in route:
            channel_state = user_channels.get(channel_id, {})
            if channel_state.get("enabled"):
                deliveries.append(
                    {
                        "message": msg_id,
                        "channel": channel_id,
                        "user": recipient,
                    }
                )
                break
    return {"deliveries": deliveries, "rejections": rejections}
