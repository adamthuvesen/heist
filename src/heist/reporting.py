from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from heist import difficulty
from heist.models import TaskRunResult
from heist.usage import primary_cost as _primary_cost

if TYPE_CHECKING:
    # heist.history imports from heist.runner which imports from this module,
    # so the runtime import is broken to avoid a cycle. ComparisonReport is
    # only used at type-check time and through duck-typed attribute access.
    from heist.history import ComparisonReport

SATURATION_THRESHOLD = 0.90


# Headline metric is alpha (α), the difficulty-weighted score; success@0.999
# solve-rate (wins / tasks) is kept as a secondary column. Ranking ties on alpha
# break by mean score so the order stays deterministic and sensible.
def _solve_rate(wins: int, tasks: int) -> float:
    return wins / tasks if tasks else 0.0


def _rank_key(a: dict[str, object]) -> tuple[float, float]:
    """Primary ranking metric: alpha (α), ties broken by mean score."""
    return (float(a["alpha"]), float(a["mean"]))


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html"

# Map heist agent_id prefix → short id used in the editorial template's CSS classes
# (.tone-opus, .swatch.composer, etc). Unknown agents get a generic slug.
_AGENT_SHORT_ID = {
    "claude-opus": "opus",
    "claude-sonnet": "sonnet",
    "claude-haiku": "haiku",
    # More specific codex prefixes must precede "codex-gpt" — _short_agent_id
    # prefix-matches in insertion order, so the mini variant needs its own id or
    # it collapses onto "codex" and the two codex agents overwrite each other.
    "codex-gpt-5.4-mini": "codexmini",
    "codex-gpt": "codex",
    "cursor-composer": "composer",
    "cursor-grok": "grok",
    "cursor-kimi": "kimi",
    "cursor-gemini": "gemini",
    "openrouter-gemini": "openrouter-gemini",
    "openrouter-deepseek": "openrouter-deepseek",
    "openrouter-kimi": "openrouter-kimi",
    "openrouter-qwen": "openrouter-qwen",
}

# Tier label per short id (used in the scoreboard column header).
_AGENT_TIER = {
    "opus": "flagship",
    "codex": "flagship",
    "codexmini": "small",
    "sonnet": "mid-tier",
    "haiku": "small",
    "composer": "cursor flagship",
    "grok": "reasoning",
    "kimi": "open-weights",
    "gemini": "flagship",
    "openrouter-gemini": "flagship",
    "openrouter-deepseek": "flagship",
    "openrouter-kimi": "open-weights",
    "openrouter-qwen": "open-weights",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _latency_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _run_date_str() -> str:
    # Avoid strftime("%-d") — the no-pad day directive is a glibc/BSD extension
    # that breaks on Windows. Build the day without it.
    dt = datetime.now()
    return f"{dt.day} {dt:%b %Y}"


_SHORT_LABEL_PREFIXES = ("Claude ", "Codex ", "Cursor ", "OpenRouter ")
_SHORT_LABEL_SUFFIX = " High"


def _short_agent_label(label: str) -> str:
    for prefix in _SHORT_LABEL_PREFIXES:
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    if label.endswith(_SHORT_LABEL_SUFFIX):
        label = label[: -len(_SHORT_LABEL_SUFFIX)]
    return label


def _render_score_chart(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []

    ranked = sorted(
        rows,
        key=lambda r: (float(r["alpha"]), float(r["mean_score"])),
        reverse=True,
    )
    bar_height = 10
    bar_width = 5
    sub_blocks = (" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
    labels = [_short_agent_label(str(row["label"])) for row in ranked]
    scores = [_pct(float(row["alpha"])) for row in ranked]
    slot_width = max(bar_width, max(len(item) for item in labels + scores))
    gap = "  "
    indent = "      "
    axis_gutter = "  0% └"

    chart_lines: list[str] = []
    for tier in range(bar_height, 0, -1):
        cells: list[str] = []
        for row in ranked:
            score = float(row["alpha"])
            eighths = max(0, min(bar_height * 8, round(score * bar_height * 8)))
            fill = eighths - (tier - 1) * 8
            fill = max(0, min(8, fill))
            cell_char = sub_blocks[fill]
            cells.append((cell_char * bar_width).center(slot_width))
        prefix = f"{int(tier * 100 / bar_height):3d}% ┤"
        chart_lines.append(prefix + gap + gap.join(cells))

    axis_width = (slot_width + len(gap)) * len(ranked) + 1
    chart_lines.append(axis_gutter + "─" * axis_width)

    score_row = indent + gap.join(scores[i].center(slot_width) for i in range(len(ranked)))
    label_row = indent + gap.join(labels[i].center(slot_width) for i in range(len(ranked)))
    chart_lines.append(score_row)
    chart_lines.append(label_row)
    return chart_lines


@dataclass
class _AgentMetrics:
    """Per-agent aggregates shared by the markdown and HTML renderers.

    Centralised so the two views can't drift in how they count wins, mean
    score, or median latency on passing tasks. Renderers project to their own
    dict shapes; this dataclass is the single source of truth for the numbers.
    """

    agent_id: str
    label: str
    model_id: str
    tasks: int
    wins: int
    ge90: int
    mean_score: float
    total_cost: float
    total_latency: float
    success_latency: float
    median_latency: float
    median_success_latency: float
    clean_pass_count: int
    clean_pass_lat_median: float
    any_cost_estimated: bool
    alpha: float


def _agent_metrics(agent_id: str, agent_results: list[TaskRunResult]) -> _AgentMetrics:
    n = len(agent_results)
    graded = [r for r in agent_results if r.outcome_status == "graded"]
    wins = sum(1 for r in graded if r.success)
    ge90 = sum(1 for r in agent_results if r.score >= 0.90)
    mean_score = sum(r.score for r in agent_results) / n if n else 0.0
    total_cost = sum(_primary_cost(r) or 0.0 for r in agent_results)
    total_latency = sum(r.latency_s or 0.0 for r in agent_results)
    all_latencies = [r.latency_s or 0.0 for r in agent_results]
    # Latency restricted to passing tasks — separates "fast model" from
    # "model that gives up sooner / thrashes on hard tasks".
    success_latencies = [r.latency_s or 0.0 for r in graded if r.success]
    success_latency = sum(success_latencies)
    # Latency restricted to passing tasks that COMPLETED normally — excludes
    # wall-clock-killed rows whose latency reflects the timeout cap rather
    # than the agent's actual speed. Gate on the timeout signal itself, not
    # cost provenance: a passing row's measured wall-clock latency is
    # trustworthy even when its *cost* could only be estimated.
    clean_pass = [r for r in agent_results if r.success and not r.timed_out]
    clean_pass_latencies = [r.latency_s or 0.0 for r in clean_pass]
    any_estimated = any(r.cost_source in ("reconstructed", "estimated") for r in agent_results)
    alpha = difficulty.sc_alpha((r.task_id, r.score) for r in agent_results)
    return _AgentMetrics(
        agent_id=agent_id,
        label=agent_results[0].agent_label,
        model_id=agent_results[0].model_id,
        tasks=n,
        wins=wins,
        ge90=ge90,
        mean_score=mean_score,
        total_cost=total_cost,
        total_latency=total_latency,
        success_latency=success_latency,
        median_latency=_latency_median(all_latencies),
        median_success_latency=_latency_median(success_latencies),
        clean_pass_count=len(clean_pass),
        clean_pass_lat_median=_latency_median(clean_pass_latencies),
        any_cost_estimated=any_estimated,
        alpha=alpha,
    )


def _compute_agent_metrics(results: list[TaskRunResult]) -> list[_AgentMetrics]:
    groups: dict[str, list[TaskRunResult]] = defaultdict(list)
    for result in results:
        groups[result.agent_id].append(result)
    return [
        _agent_metrics(agent_id, agent_results)
        for agent_id, agent_results in sorted(groups.items())
    ]


def summarize_by_agent(results: list[TaskRunResult]) -> list[dict[str, object]]:
    """Markdown-renderer projection of per-agent metrics."""
    return [
        {
            "agent_id": m.agent_id,
            "label": m.label,
            "model_id": m.model_id,
            "tasks": m.tasks,
            "successes": m.wins,
            "success_rate": m.wins / m.tasks if m.tasks else 0.0,
            "mean_score": m.mean_score,
            "alpha": m.alpha,
            "total_cost": m.total_cost,
            "total_latency": m.total_latency,
            "success_latency": m.success_latency,
            "median_success_latency": m.median_success_latency,
            "saturated": (m.wins / m.tasks if m.tasks else 0.0) >= SATURATION_THRESHOLD,
        }
        for m in _compute_agent_metrics(results)
    ]


def render_markdown(results: list[TaskRunResult]) -> str:
    rows = summarize_by_agent(results)
    # Lead with the headline metric: rank agents by alpha (α), breaking ties on
    # mean score.
    rows.sort(key=lambda r: (float(r["alpha"]), float(r["mean_score"])), reverse=True)
    saturated = [row for row in rows if row["saturated"]]
    lines = [
        "# HEIST Run Report",
        "",
        "## Summary",
        "",
        "Ranked by alpha (α): each task is difficulty-weighted — hard tasks "
        "count 3×, easy ones half.",
        "",
        "| Agent | Model | Tasks | alpha |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | `{model}` | {tasks} | {alpha} |".format(
                label=row["label"],
                model=row["model_id"],
                tasks=row["tasks"],
                alpha=_pct(float(row["alpha"])),
            )
        )

    chart_lines = _render_score_chart(rows)
    if chart_lines:
        lines.extend(["", "## alpha Ranking", "", "```text"])
        lines.extend(chart_lines)
        lines.append("```")

    lines.extend(["", "## Hardness Gate", ""])
    if saturated:
        names = ", ".join(str(row["label"]) for row in saturated)
        lines.append(
            f"SATURATED: {names} reached >= {_pct(SATURATION_THRESHOLD)} success. "
            "Do not trust this suite for frontier model ranking until harder tasks are added."
        )
    else:
        lines.append(f"Not saturated: no agent reached {_pct(SATURATION_THRESHOLD)} success.")

    lines.append("")
    return "\n".join(lines)


def _esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _short_agent_id(agent_id: str) -> str:
    """Map heist agent_id to the short id used in editorial CSS classes."""
    for prefix, short in _AGENT_SHORT_ID.items():
        if agent_id.startswith(prefix):
            return short
    return agent_id.split("-")[0]


def _build_agent_summary(results: list[TaskRunResult]) -> list[dict[str, object]]:
    """HTML-renderer projection of per-agent metrics, with short_id and tier."""
    summary: list[dict[str, object]] = []
    for m in _compute_agent_metrics(results):
        short = _short_agent_id(m.agent_id)
        # Note: the HTML view counts "wins" off the .success flag rather than
        # restricting to graded rows; results with success=True are by
        # definition graded, so the two counts agree.
        summary.append(
            {
                "agent_id": m.agent_id,
                "short_id": short,
                "label": m.label,
                "model_id": m.model_id,
                "tier": _AGENT_TIER.get(short, ""),
                "n": m.tasks,
                "wins": m.wins,
                "ge90": m.ge90,
                "mean": m.mean_score,
                "alpha": m.alpha,
                "cost": m.total_cost,
                "cost_est": m.any_cost_estimated,
                "lat_median": m.median_latency,
                "lat_success_median": m.median_success_latency,
                "lat_clean_pass_median": m.clean_pass_lat_median,
            }
        )
    return summary


def _format_abstract_cells(summary: list[dict[str, object]], n_no_pass: int, n_tasks: int) -> str:
    """Build the abstract grid: alpha leader, ≥90% leader, ceiling."""
    by_alpha = max(summary, key=_rank_key)
    by_ge90 = max(summary, key=lambda a: (a["ge90"], a["mean"]))

    cells: list[str] = []

    def cell(k: str, v: str, unit: str) -> str:
        return (
            f'<div class="cell"><span class="k">{k}</span>'
            f'<span class="v">{v}</span>'
            f'<span class="unit">{unit}</span></div>'
        )

    cells.append(
        cell(
            "Highest alpha",
            _esc(by_alpha["label"]),
            f"α {_pct(float(by_alpha['alpha']))}",
        )
    )
    cells.append(
        cell(
            "Most ≥ 90% tasks",
            _esc(by_ge90["label"]),
            f"{by_ge90['ge90']} of {by_ge90['n']} at ≥ 90%",
        )
    )
    cells.append(
        cell(
            "Tasks no agent passed",
            str(n_no_pass),
            # n_no_pass is counted over the union of all task_ids; the
            # denominator must be that same distinct task count, not the first
            # agent's task count (agents can run different subsets).
            f"of {n_tasks} unsolved",
        )
    )
    return "".join(cells)


def _format_lede(summary: list[dict[str, object]], n_tasks: int) -> str:
    """One-sentence lede: alpha leader."""
    by_alpha = max(summary, key=_rank_key)
    alpha_tone = by_alpha["short_id"]
    return (
        f"{n_tasks} tasks. Each task is a real multi-file repo with a hidden grader. "
        f'<em class="tone-{alpha_tone}">{_esc(by_alpha["label"])}</em> leads on alpha, '
        f"α {_pct(float(by_alpha['alpha']))}."
    )


def _format_narrative_blocks(summary: list[dict[str, object]]) -> str:
    """One <div class="observation"> per agent, ranked by solve-rate, with auto-derived facts."""
    ranked = sorted(summary, key=_rank_key, reverse=True)
    # Identify rank-based callouts. "Fastest" uses success-only median latency so
    # a model that gives up quickly on hard tasks doesn't get crowned the speed
    # leader; it is stated as a relative claim, no absolute time shown.
    min_lat = min(
        (a["lat_success_median"] for a in summary if a["lat_success_median"] > 0),
        default=0.0,
    )
    max_wins = max(a["wins"] for a in summary)
    max_alpha = max(float(a["alpha"]) for a in summary)

    blocks: list[str] = []
    for a in ranked:
        tone = a["short_id"]
        tone_class = f" {tone}-tone" if tone in _AGENT_TIER else ""
        callouts: list[str] = []
        if abs(float(a["alpha"]) - max_alpha) < 1e-9:
            callouts.append("alpha leader")
        if a["wins"] == max_wins:
            callouts.append("Most solved")
        # Crown only the strict minimum (float-equality epsilon, like the alpha
        # leader above) — a tolerance band would let several near-ties all claim
        # "Fastest".
        if min_lat > 0 and abs(float(a["lat_success_median"]) - min_lat) < 1e-9:
            callouts.append("Fastest on passing tasks")

        base = (
            f"α {_pct(float(a['alpha']))} · mean {_pct(float(a['mean']))} · "
            f"{a['wins']}/{a['n']} solved · {a['ge90']} at ≥ 90%."
        )
        callout_str = (" " + " · ".join(callouts)) if callouts else ""
        blocks.append(
            f'<div class="observation{tone_class}">'
            f"<h4>{_esc(_short_agent_label(str(a['label'])))}</h4>"
            f"<p>{base}{callout_str}</p>"
            "</div>"
        )
    return "".join(blocks)


def _build_run_json(
    results: list[TaskRunResult],
    summary: list[dict[str, object]],
    n_no_pass: int,
    total_cost: float,
) -> str:
    """Serialize window.RUN — agents/pairs/tasks/global — exactly as the editorial JS expects."""
    agents_data = []
    rank_order = sorted(summary, key=_rank_key, reverse=True)
    for a in rank_order:
        entry = {
            "id": a["short_id"],
            "label": a["label"],
            "tier": a["tier"],
            "n": a["n"],
            "wins": a["wins"],
            # Secondary column: success@0.999 solve-rate (wins / tasks); the
            # headline metric is alpha (α), below.
            "solve": round(_solve_rate(int(a["wins"]), int(a["n"])), 4),
            "ge90": a["ge90"],
            "mean": a["mean"],
            "alpha": round(float(a["alpha"]), 4),
            "cost": round(float(a["cost"]), 4),
            # Speed signal: median latency on tasks that PASSED and completed
            # normally (excludes wall-clock-killed rows whose latency reflects
            # the timeout cap, not the agent's actual speed). Fall back through
            # success-only median, then overall median, when an agent has no
            # clean passes (rare, but keeps the chart from blanking).
            "lat": round(
                float(a["lat_clean_pass_median"] or a["lat_success_median"] or a["lat_median"]),
                2,
            ),
        }
        if a["cost_est"]:
            entry["cost_est"] = True
        agents_data.append(entry)

    pairs = []
    for r in sorted(results, key=lambda x: (x.agent_id, x.task_id)):
        short = _short_agent_id(r.agent_id)
        cost = _primary_cost(r) or 0
        pairs.append(
            {
                "agent": short,
                "task": r.task_id,
                "score": round(r.score, 4),
                "latency_s": round(r.latency_s or 0.0, 4),
                "cost_usd": round(float(cost), 4),
                "success": r.success,
            }
        )

    # Group once instead of filtering `results` per task — the previous form
    # was O(tasks × results), quadratic with suite size.
    by_task: dict[str, list[TaskRunResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)
    tasks = sorted(by_task)
    n_tasks = len(tasks)
    n_agents = len(summary)
    all_perfect = sum(1 for rs in by_task.values() if all(r.score == 1.0 for r in rs))
    all_ge90 = sum(1 for rs in by_task.values() if all(r.score >= 0.90 for r in rs))

    data = {
        "agents": agents_data,
        "rank_order_ids": [a["short_id"] for a in rank_order],
        "pairs": pairs,
        "tasks": tasks,
        "global": {
            "n_tasks": n_tasks,
            "n_agents": n_agents,
            "all_perfect": all_perfect,
            "all_ge90": all_ge90,
            "nobody_passes": n_no_pass,
            "total_cost": round(total_cost, 4),
        },
    }
    return json.dumps(data, separators=(",", ":"))


def _format_score_delta_cell(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    klass = "delta-pos" if value >= 0 else "delta-neg"
    return f'<span class="{klass}">{sign}{abs(value) * 100:.1f}pp</span>'


def _format_cost_delta_cell(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "−"
    klass = "delta-pos" if value <= 0 else "delta-neg"
    return f'<span class="{klass}">{sign}${abs(value):.4f}</span>'


def _format_latency_delta_cell(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "−"
    klass = "delta-pos" if value <= 0 else "delta-neg"
    return f'<span class="{klass}">{sign}{abs(value):.1f}s</span>'


def _render_baseline_section(report: ComparisonReport) -> str:
    """Render the `--compare-baseline` HTML block, slotted into {{BASELINE_SECTION}}."""
    parts: list[str] = ['<section class="reveal baseline-section">']
    parts.append('<div class="section-head">')
    parts.append('<span class="folio">§ 09</span>')
    parts.append('<h2 class="section-title">Versus baseline</h2>')
    parts.append('<span class="section-kicker">delta vs prior run</span>')
    parts.append("</div>")

    parts.append('<div class="baseline-meta">')
    parts.append(
        f"Baseline <b>{_esc(report.run_a.run_id)}</b> "
        f"({_esc(report.run_a.kind)}, suite={_esc(report.run_a.suite)}, "
        f"sha={_esc((report.run_a.harness_git_sha or '—')[:12])})"
        " vs current "
        f"<b>{_esc(report.run_b.run_id)}</b> "
        f"({_esc(report.run_b.kind)}, suite={_esc(report.run_b.suite)}, "
        f"sha={_esc((report.run_b.harness_git_sha or '—')[:12])})."
    )
    parts.append("</div>")

    if report.harness_drift:
        parts.append(f'<div class="drift-banner">⚠ {_esc(report.harness_drift)}</div>')

    if not report.rows:
        parts.append(
            '<p class="add-remove">No shared (agent, task) pairs between '
            "baseline and current run.</p>"
        )
    else:
        by_agent: dict[str, list] = defaultdict(list)
        for row in report.rows:
            by_agent[row.agent_id].append(row)
        for agent_id in sorted(by_agent):
            parts.append(f'<h3 class="baseline-agent">{_esc(agent_id)}</h3>')
            parts.append('<table class="baseline-table">')
            parts.append(
                "<thead><tr>"
                "<th>Task</th>"
                "<th>Score (baseline)</th>"
                "<th>Score (current)</th>"
                "<th>Δ score</th>"
                "<th>Δ latency</th>"
                "<th>Δ cost</th>"
                "<th>Note</th>"
                "</tr></thead><tbody>"
            )
            for row in by_agent[agent_id]:
                row_class = ' class="regression"' if row.regression else ""
                if row.regression == "pass_to_fail":
                    note = "pass → fail"
                elif row.regression == "score_drop":
                    note = "score drop > 10pp"
                elif row.outcome_status_a != row.outcome_status_b:
                    note = f"{row.outcome_status_a} → {row.outcome_status_b}"
                else:
                    note = ""
                parts.append(
                    f"<tr{row_class}>"
                    f"<td>{_esc(row.task_id)}</td>"
                    f'<td class="num">{_pct(row.score_a)}</td>'
                    f'<td class="num">{_pct(row.score_b)}</td>'
                    f'<td class="num">{_format_score_delta_cell(row.delta_score)}</td>'
                    f'<td class="num">{_format_latency_delta_cell(row.delta_latency_s)}</td>'
                    f'<td class="num">{_format_cost_delta_cell(row.delta_cost_usd)}</td>'
                    f"<td>{_esc(note)}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")

    if report.tasks_only_in_a:
        parts.append(
            '<p class="add-remove">Tasks only in baseline: '
            f"{_esc(', '.join(report.tasks_only_in_a))}.</p>"
        )
    if report.tasks_only_in_b:
        parts.append(
            '<p class="add-remove">Tasks only in current: '
            f"{_esc(', '.join(report.tasks_only_in_b))}.</p>"
        )
    if report.agents_only_in_a:
        parts.append(
            '<p class="add-remove">Agents only in baseline: '
            f"{_esc(', '.join(report.agents_only_in_a))}.</p>"
        )
    if report.agents_only_in_b:
        parts.append(
            '<p class="add-remove">Agents only in current: '
            f"{_esc(', '.join(report.agents_only_in_b))}.</p>"
        )

    parts.append("</section>")
    return "".join(parts)


def _render_empty_report(template: str) -> str:
    """Minimal report for a run that produced zero results (e.g. fail-fast
    aborted everything before any row landed). Avoids `max()` on empty
    sequences inside the regular substitution path."""
    empty_substitutions = {
        "{{TITLE}}": "HEIST Run — No Results",
        "{{MAST_LEFT}}": "HEIST",
        "{{RUN_DATE}}": _run_date_str(),
        "{{MAST_TITLE}}": "No results.<br>Run produced <em>no rows.</em>",
        "{{AGENT_LIST}}": "",
        "{{N_TASKS}}": "0",
        "{{N_TASKS_WORD}}": "0",
        "{{TASK_COUNT_LINE}}": "0 tasks graded",
        "{{CATEGORY_LINE}}": "",
        "{{LEDE}}": "<p>No tasks completed in this run.</p>",
        "{{ABSTRACT_CELLS}}": "",
        "{{NARRATIVE_BLOCKS}}": "",
        "{{BASELINE_SECTION}}": "",
        "{{KIND_BADGE}}": "",
        "{{FOOTER_LEAD}}": "HEIST / empty run",
        "{{RUN_JSON}}": "{}",
    }
    html = template
    for token, value in empty_substitutions.items():
        html = html.replace(token, value)
    return html


def _json_for_script_tag(payload: str) -> str:
    """Defensive: prevent any string in the JSON from closing the <script> tag
    early. JSON `</script>` inside a value would otherwise terminate the block
    and let arbitrary HTML follow. The escapes are valid JSON for U+003C / U+003E."""
    return (
        payload.replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _render_kind_badge(replay_source_run_id: str | None) -> str:
    if replay_source_run_id is None:
        return ""
    return (
        '<div class="kind-badge">Replay · '
        f"source <b>{_esc(replay_source_run_id)}</b> · "
        "agents not measured</div>"
    )


def render_html(
    results: list[TaskRunResult],
    *,
    baseline_comparison: ComparisonReport | None = None,
    replay_source_run_id: str | None = None,
) -> str:
    summary = _build_agent_summary(results)

    template = _TEMPLATE_PATH.read_text()
    if not summary:
        return _render_empty_report(template)

    n_agents = len(summary)
    tasks = sorted({r.task_id for r in results})
    n_tasks = len(tasks)
    total_cost = sum(float(a["cost"]) for a in summary)

    # Tasks where no agent succeeded.
    by_task: dict[str, list[TaskRunResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)
    n_no_pass = sum(1 for rs in by_task.values() if not any(r.success for r in rs))

    substitutions = {
        "{{TITLE}}": _esc(f"HEIST Frontier — {n_agents}-agent run"),
        "{{MAST_LEFT}}": "HEIST · Frontier Suite",
        "{{RUN_DATE}}": _run_date_str(),
        "{{KIND_BADGE}}": _render_kind_badge(replay_source_run_id),
        "{{MAST_TITLE}}": "<em>HEIST</em>",
        "{{AGENT_LIST}}": " · ".join(
            _esc(_short_agent_label(str(a["label"])))
            for a in sorted(summary, key=_rank_key, reverse=True)
        ),
        "{{N_TASKS}}": str(n_tasks),
        "{{N_TASKS_WORD}}": str(n_tasks),
        "{{TASK_COUNT_LINE}}": f"{n_tasks} hidden-grader · multi-file",
        "{{CATEGORY_LINE}}": "Repo-debugging frontier",
        "{{LEDE}}": _format_lede(summary, n_tasks),
        "{{ABSTRACT_CELLS}}": _format_abstract_cells(summary, n_no_pass, n_tasks),
        "{{NARRATIVE_BLOCKS}}": _format_narrative_blocks(summary),
        "{{BASELINE_SECTION}}": (
            _render_baseline_section(baseline_comparison) if baseline_comparison is not None else ""
        ),
        "{{FOOTER_LEAD}}": (f"HEIST / frontier / <b>{n_tasks}-task · {n_agents}-agent run</b>"),
        "{{RUN_JSON}}": _json_for_script_tag(
            _build_run_json(results, summary, n_no_pass, total_cost)
        ),
    }

    html = template
    for token, value in substitutions.items():
        html = html.replace(token, value)
    return html


def write_report(
    run_dir: Path,
    results: list[TaskRunResult],
    *,
    baseline_comparison: ComparisonReport | None = None,
    replay_source_run_id: str | None = None,
) -> Path:
    md_path = run_dir / "summary.md"
    md_path.write_text(render_markdown(results))
    html_path = run_dir / "report.html"
    html_path.write_text(
        render_html(
            results,
            baseline_comparison=baseline_comparison,
            replay_source_run_id=replay_source_run_id,
        )
    )
    return md_path
