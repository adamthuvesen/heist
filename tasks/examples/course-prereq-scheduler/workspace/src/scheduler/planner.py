from __future__ import annotations

from scheduler.graph import find_cycle_members  # noqa: F401  (helper available, unused by seed)


def plan_schedule(
    courses: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    max_q = int(config.get("max_quarters", 8))
    remaining: dict[str, dict[str, object]] = {str(c["id"]): c for c in courses}
    completed: set[str] = set()
    schedule: list[list[str]] = []

    for _ in range(max_q):
        ready_ids = sorted(
            cid
            for cid, c in remaining.items()
            if all(p in completed for p in c.get("prereqs", []) or [])
        )
        if not ready_ids:
            break
        schedule.append(ready_ids)
        for cid in ready_ids:
            completed.add(cid)
            del remaining[cid]

    return {
        "schedule": schedule,
        "deferred": sorted(remaining.keys()),
        "rejections": {},
    }
