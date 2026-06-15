from __future__ import annotations

from scheduler.graph import find_cycle_members


def plan_schedule(
    courses: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    cap = int(config.get("max_credits_per_quarter", 0))
    max_q = int(config.get("max_quarters", 0))

    by_id: dict[str, dict[str, object]] = {str(c["id"]): c for c in courses}
    rejections: dict[str, str] = {}

    # Overweight: course's own credits exceed the per-quarter cap.
    for cid, course in by_id.items():
        if int(course.get("credits", 0)) > cap:
            rejections[cid] = "overweight"

    # Missing prereq: any prereq id not present in the course set.
    for cid, course in by_id.items():
        if cid in rejections:
            continue
        for prereq in course.get("prereqs", []) or []:
            if prereq not in by_id:
                rejections[cid] = "missing_prereq"
                break

    # Cycles: build the prereq graph over remaining schedulable courses and
    # let the helper find cycle members. Courses that depend on errored
    # prereqs (overweight, missing) end up in `deferred` via the natural
    # readiness check — no cascade tracking needed.
    schedulable = {
        cid: [p for p in (course.get("prereqs", []) or []) if p in by_id]
        for cid, course in by_id.items()
        if cid not in rejections
    }
    for cid in find_cycle_members(schedulable):
        rejections[cid] = "cycle"

    # Pack the remaining courses quarter by quarter.
    remaining = {cid: by_id[cid] for cid in by_id if cid not in rejections}
    completed: set[str] = set()
    schedule: list[list[str]] = []

    for _ in range(max_q):
        ready = sorted(
            cid
            for cid, course in remaining.items()
            if all(p in completed for p in (course.get("prereqs", []) or []))
        )
        if not ready:
            break
        quarter: list[str] = []
        used = 0
        for cid in ready:
            credits = int(remaining[cid].get("credits", 0))
            if used + credits <= cap:
                quarter.append(cid)
                used += credits
        if not quarter:
            break
        schedule.append(quarter)
        for cid in quarter:
            completed.add(cid)
            del remaining[cid]

    return {
        "schedule": schedule,
        "deferred": sorted(remaining.keys()),
        "rejections": rejections,
    }
