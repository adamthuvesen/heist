"""Regenerate the public README figures from the sanitized report payload.

The public repo does not commit ``leaderboard.jsonl`` or raw ``runs/`` output.
Figures are redrawn from the aggregate-only ``window.RUN`` payload embedded in
``results/frontier-2026-06-15/report.html``:

    results/frontier-2026-06-15/img/scoreboard.png
    results/frontier-2026-06-15/img/performance-frontier.png

That payload contains only per-agent alpha and relative cost/speed indices. It
has no per-task rows, absolute spend, latencies, local paths, or held-out task
ids.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAGE = "#f1ece0"
INK = "#2b2620"
MUTED = "#6f685b"
GRID = "#d9d1c0"


class Agent(TypedDict):
    id: str
    label: str
    color: str
    alpha: float
    cost: float
    lat: float


def _load_agents(report_path: Path) -> list[Agent]:
    text = report_path.read_text()
    match = re.search(r"window\.RUN = (\{.*?\});", text)
    if not match:
        raise SystemExit(f"{report_path}: could not find window.RUN payload")
    payload = json.loads(match.group(1))
    agents = payload.get("agents")
    order = payload.get("rank_order_ids")
    if not isinstance(agents, list) or not isinstance(order, list):
        raise SystemExit(f"{report_path}: payload is missing agents/rank_order_ids")
    by_id = {agent["id"]: agent for agent in agents}
    missing = [agent_id for agent_id in order if agent_id not in by_id]
    if missing:
        raise SystemExit(f"{report_path}: rank_order_ids references missing agents: {missing}")
    return [by_id[agent_id] for agent_id in order]


def _pct(value: float) -> float:
    return value * 100


def draw_scoreboard(agents: list[Agent], out_path: Path) -> None:
    if not agents:
        raise SystemExit("make_figures: no agents to plot in scoreboard")
    fig, ax = plt.subplots(figsize=(12.74, 7.61), facecolor=PAGE)
    ax.set_facecolor(PAGE)

    positions = range(len(agents))
    for i, agent in zip(positions, agents, strict=True):
        alpha = _pct(float(agent["alpha"]))
        color = str(agent["color"])
        ax.bar(i, alpha, width=0.62, color=color, zorder=3)
        ax.text(
            i,
            alpha + 1.4,
            f"{alpha:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax.text(i, -6.2, str(agent["label"]), ha="center", va="top", fontsize=9.5, color=INK)
        ax.text(i, -10.8, f"No {i + 1}", ha="center", va="top", fontsize=8.5, color=MUTED)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.7, len(agents) - 0.3)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=MUTED)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(
        "Alpha (α) · ranked",
        loc="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
        pad=22,
    )
    ax.text(
        1.0,
        1.032,
        "HIGHEST TO THE LEFT",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED,
        fontfamily="monospace",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.18)
    fig.savefig(out_path, dpi=100, facecolor=PAGE)
    plt.close(fig)


def _label_y(agents: list[Agent]) -> dict[str, float]:
    values = [_pct(float(agent["alpha"])) for agent in agents]
    span = max(values) - min(values)
    min_gap = span * 0.07 + 0.6
    placed: dict[str, float] = {}
    last: float | None = None
    for agent in sorted(agents, key=lambda item: float(item["alpha"])):
        y = _pct(float(agent["alpha"]))
        if last is not None and y - last < min_gap:
            y = last + min_gap
        placed[str(agent["id"])] = y
        last = y
    return placed


def _scatter_panel(
    ax,
    agents: list[Agent],
    field: str,
    title: str,
    xlabel: str,
    corner: str,
    *,
    log_x: bool,
) -> None:
    ax.set_facecolor(PAGE)
    xs = [float(agent[field]) for agent in agents]
    if any(value <= 0 or not math.isfinite(value) for value in xs):
        raise SystemExit(f"make_figures: {field} index contains non-positive or non-finite values")

    label_y = _label_y(agents)
    pivot = (min(xs) * max(xs)) ** 0.5
    for agent in agents:
        x = float(agent[field])
        y = _pct(float(agent["alpha"]))
        color = str(agent["color"])
        ax.scatter(x, y, s=170, color=color, edgecolor=PAGE, linewidth=1.3, zorder=3)
        right = x < pivot
        ax.annotate(
            str(agent["label"]),
            (x, label_y[str(agent["id"])]),
            xytext=(8 if right else -8, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=color,
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    else:
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=16)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED, fontfamily="monospace")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.text(
        1.0,
        1.01,
        corner,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=MUTED,
        fontfamily="monospace",
    )


def draw_performance_frontier(agents: list[Agent], out_path: Path) -> None:
    if not agents:
        raise SystemExit("make_figures: no agents to plot in performance frontier")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.51, 6.64), facecolor=PAGE)
    fig.suptitle(
        "Performance frontier",
        x=0.06,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )

    _scatter_panel(
        left,
        agents,
        "cost",
        "Score × Cost",
        "COST INDEX · CHEAPEST = 1x",
        "↑ BETTER · ← CHEAPER",
        log_x=True,
    )
    left.set_ylabel("alpha (%)", fontsize=9, color=MUTED, fontfamily="monospace")
    _scatter_panel(
        right,
        agents,
        "lat",
        "Score × Speed",
        "SPEED INDEX · FASTEST = 1x",
        "↑ BETTER · ← FASTER",
        log_x=False,
    )

    scores = [_pct(float(agent["alpha"])) for agent in agents]
    pad = (max(scores) - min(scores)) * 0.18 + 1
    for ax in (left, right):
        ax.set_ylim(min(scores) - pad, max(scores) + pad)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.text(
        0.06,
        0.025,
        "Both axes are indexed to the best in class (cheapest = 1x, fastest = 1x).",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )
    fig.subplots_adjust(left=0.065, right=0.975, top=0.83, bottom=0.14, wspace=0.2)
    fig.savefig(out_path, dpi=100, facecolor=PAGE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Redraw public README figures from report.html.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/frontier-2026-06-15/report.html"),
        help="public report.html containing sanitized window.RUN",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/frontier-2026-06-15/img"),
        help="directory to write scoreboard.png and performance-frontier.png",
    )
    args = parser.parse_args()

    agents = _load_agents(args.report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw_scoreboard(agents, args.out_dir / "scoreboard.png")
    draw_performance_frontier(agents, args.out_dir / "performance-frontier.png")
    print(f"wrote {args.out_dir}/scoreboard.png and {args.out_dir}/performance-frontier.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
