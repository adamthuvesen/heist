from __future__ import annotations


def find_cycle_members(prereqs: dict[str, list[str]]) -> set[str]:
    """Return the set of course ids that cannot be topologically ordered
    within the given subgraph — i.e., those participating in a cycle.

    Uses Kahn's algorithm: any node whose indegree never reaches zero
    within ``prereqs`` is returned. Edges pointing to ids not in
    ``prereqs`` are ignored; callers must handle missing-prereq and
    upstream-error cascades separately.
    """
    indegree: dict[str, int] = {}
    for node, deps in prereqs.items():
        indegree.setdefault(node, 0)
        for dep in deps:
            if dep in prereqs:
                indegree[node] = indegree.get(node, 0) + 1

    ready = sorted(node for node, deg in indegree.items() if deg == 0)
    resolved: set[str] = set()
    while ready:
        node = ready.pop(0)
        resolved.add(node)
        for other, deps in prereqs.items():
            if node in deps and other in indegree and other not in resolved:
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
        ready.sort()
    return set(prereqs.keys()) - resolved
