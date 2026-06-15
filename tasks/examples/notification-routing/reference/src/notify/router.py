from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _parse_offset(tz: str) -> timezone:
    sign = 1 if tz.startswith("+") else -1
    hours, minutes = tz[1:].split(":")
    return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))


def _local_hhmm(sent_at: str, tz: str) -> str:
    dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
    return dt.astimezone(_parse_offset(tz)).strftime("%H:%M")


def _in_dnd(local: str, start: str, end: str) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= local < end
    return local >= start or local < end


def _match_rule(rule: dict, message: dict) -> bool:
    match = rule.get("match", {}) or {}
    if "type" in match and match["type"] != message.get("type"):
        return False
    return not ("priority" in match and match["priority"] != message.get("priority"))


def _try_channels(
    channel_ids: list,
    user_channels: dict,
    channels: dict,
    message_type: object,
    dnd_blocked: bool,
) -> tuple[str | None, bool]:
    saw_dnd_block = False
    for channel_id in channel_ids:
        channel_state = user_channels.get(channel_id, {})
        if not channel_state.get("enabled"):
            continue
        channel_def = channels.get(channel_id, {})
        capabilities = channel_def.get("capabilities", []) or []
        if message_type not in capabilities:
            continue
        if dnd_blocked:
            saw_dnd_block = True
            continue
        return channel_id, saw_dnd_block
    return None, saw_dnd_block


def route_notifications(
    messages: list[dict[str, object]],
    users: dict[str, dict[str, object]],
    channels: dict[str, dict[str, object]],
    rules: list[dict[str, object]],
    now: str,
) -> dict[str, object]:
    seen: set[str] = set()
    deliveries: list[dict[str, object]] = []
    rejections: dict[str, str] = {}

    for message in messages:
        msg_id = str(message.get("id", ""))
        if msg_id in seen:
            rejections[msg_id] = "duplicate"
            continue
        seen.add(msg_id)

        recipient = str(message.get("recipient", ""))
        user = users.get(recipient)
        if user is None:
            rejections[msg_id] = "user_unknown"
            continue

        matched = next((r for r in rules if _match_rule(r, message)), None)
        if matched is None:
            rejections[msg_id] = "no_rule"
            continue

        priority = message.get("priority")
        dnd = user.get("dnd")
        dnd_blocked = False
        if dnd and priority != "urgent":
            tz = str(user.get("timezone", "+00:00"))
            local = _local_hhmm(str(message.get("sent_at", "")), tz)
            dnd_blocked = _in_dnd(local, str(dnd.get("start", "")), str(dnd.get("end", "")))

        user_channels = user.get("channels", {}) or {}
        msg_type = message.get("type")

        chosen, saw_dnd = _try_channels(
            matched.get("route", []) or [], user_channels, channels, msg_type, dnd_blocked
        )
        if chosen is None:
            chosen, saw_dnd_fb = _try_channels(
                matched.get("fallback", []) or [], user_channels, channels, msg_type, dnd_blocked
            )
            saw_dnd = saw_dnd or saw_dnd_fb

        if chosen is not None:
            deliveries.append({"message": msg_id, "channel": chosen, "user": recipient})
        elif dnd_blocked and saw_dnd:
            rejections[msg_id] = "dnd"
        else:
            rejections[msg_id] = "no_capable_channel"

    deliveries.sort(
        key=lambda d: (
            next((m["sent_at"] for m in messages if str(m.get("id", "")) == d["message"]), ""),
            d["message"],
        )
    )
    return {"deliveries": deliveries, "rejections": rejections}
