"""Regenerate the README figures from a published leaderboard.jsonl.

Draws the two figures the README links:

    figures/alpha-ranking.jpg        alpha (α) per agent, ranked
    figures/performance-frontier.jpg score x cost and score x speed frontiers

Everything is derived from the eight leak-free columns in leaderboard.jsonl, so
the figures are reproducible from committed data with no run dir, API keys, or
hidden graders. The goal is a faithful redraw, not a pixel match.

    uv run --group figures python scripts/make_figures.py
    make figures
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

# Allow `python scripts/make_figures.py` to import the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heist.difficulty import sc_alpha  # noqa: E402

# Parchment palette, lifted from the original figures.
PAGE = "#f1ece0"
INK = "#2b2620"
MUTED = "#6f685b"
GRID = "#d9d1c0"

# Per-agent accent colour, keyed by the short display label.
AGENT_COLORS = {
    "Claude Opus 4.7": "#bd5b36",
    "Codex GPT-5.5": "#2f7d6e",
    "Composer 2": "#3f6fa3",
    "Claude Sonnet 4.6": "#7c3a55",
    "Gemini 3.1 Pro": "#8089b8",
    "Claude Haiku 4.5": "#b58a2f",
    "Kimi K2.5": "#6f8f43",
    "Grok 4.3": "#7c6347",
}
_FALLBACK = ["#bd5b36", "#2f7d6e", "#3f6fa3", "#7c3a55", "#8089b8", "#b58a2f", "#6f8f43", "#7c6347"]


def short_label(agent: str) -> str:
    """Trim the verbose registry label to the compact name used on the figures."""
    label = agent.removeprefix("Cursor ").removesuffix(" High")
    return label.strip()


class AgentStat:
    def __init__(self, label: str) -> None:
        self.label = label
        self.scores: list[float] = []
        self.costs: list[float] = []
        self.pass_latencies: list[float] = []
        self.wins = 0

    @property
    def mean_score(self) -> float:
        return 100 * sum(self.scores) / len(self.scores)

    @property
    def alpha(self) -> float:
        # Headline metric: alpha (α), the difficulty-weighted mean score, as %.
        # The leak-free leaderboard carries no per-task tiers, so every weight
        # defaults to medium and alpha equals the mean score.
        pairs = ((None, s) for s in self.scores)
        return 100 * sc_alpha(pairs)

    @property
    def total_cost(self) -> float:
        return sum(self.costs)

    @property
    def median_pass_latency(self) -> float:
        # Speed is judged on passing runs only, so an agent is not rewarded for
        # bailing out early.
        if not self.pass_latencies:
            return float("nan")
        return statistics.median(self.pass_latencies)


def load_stats(leaderboard_path: Path) -> list[AgentStat]:
    by_agent: dict[str, AgentStat] = {}
    order: list[str] = []
    for raw in leaderboard_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        label = short_label(row["agent"])
        stat = by_agent.get(label)
        if stat is None:
            stat = by_agent[label] = AgentStat(label)
            order.append(label)
        stat.scores.append(float(row["score"]))
        stat.costs.append(float(row["cost_usd"]))
        if row["success"]:
            stat.wins += 1
            stat.pass_latencies.append(float(row["latency_s"]))
    return [by_agent[label] for label in order]


def color_for(label: str, index: int) -> str:
    return AGENT_COLORS.get(label, _FALLBACK[index % len(_FALLBACK)])


def draw_alpha_ranking(stats: list[AgentStat], out_path: Path) -> None:
    ranked = sorted(stats, key=lambda s: (s.alpha, s.wins), reverse=True)
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=PAGE)
    ax.set_facecolor(PAGE)

    positions = range(len(ranked))
    for i, stat in zip(positions, ranked, strict=True):
        color = color_for(stat.label, i)
        ax.bar(i, stat.alpha, width=0.62, color=color, zorder=3)
        ax.text(
            i,
            stat.alpha + 1.5,
            f"{stat.alpha:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        ax.text(i, -7, stat.label, ha="center", va="top", fontsize=11, color=INK)
        ax.text(i, -12.5, f"No {i + 1}", ha="center", va="top", fontsize=9, color=MUTED)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.7, len(ranked) - 0.3)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=MUTED)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    ax.set_title(
        "Alpha (α) · ranked", loc="left", fontsize=20, fontweight="bold", color=INK, pad=24
    )
    ax.text(
        1.0,
        1.035,
        "HIGHEST TO THE LEFT",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=MUTED,
        fontfamily="monospace",
    )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.16)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)


def _declutter_y(stats: list[AgentStat]) -> dict[str, float]:
    """Nudge near-tied score labels apart so they don't overlap horizontally."""
    span = max(s.mean_score for s in stats) - min(s.mean_score for s in stats)
    min_gap = span * 0.07 + 0.6
    label_y: dict[str, float] = {}
    last = None
    for stat in sorted(stats, key=lambda s: s.mean_score):
        y = stat.mean_score
        if last is not None and y - last < min_gap:
            y = last + min_gap
        label_y[stat.label] = y
        last = y
    return label_y


def _scatter_panel(ax, stats, xs, label_y, xlabel, corner, *, log_x):
    ax.set_facecolor(PAGE)
    xmax = max(xs.values())
    pivot = (min(xs.values()) * xmax) ** 0.5  # geometric midpoint: label side flips here
    for i, stat in enumerate(stats):
        color = color_for(stat.label, i)
        x = xs[stat.label]
        ax.scatter(x, stat.mean_score, s=190, color=color, edgecolor=PAGE, linewidth=1.5, zorder=3)
        right = x < pivot
        ax.annotate(
            stat.label,
            (x, label_y[stat.label]),
            xytext=(9 if right else -9, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=color,
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    else:
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED, fontfamily="monospace")
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
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
        fontsize=10,
        color=MUTED,
        fontfamily="monospace",
    )


def draw_performance_frontier(stats: list[AgentStat], out_path: Path) -> None:
    min_cost = min(s.total_cost for s in stats)
    min_lat = min(s.median_pass_latency for s in stats)
    cost_index = {s.label: s.total_cost / min_cost for s in stats}
    speed_index = {s.label: s.median_pass_latency / min_lat for s in stats}

    label_y = _declutter_y(stats)

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=PAGE)
    fig.suptitle(
        "Performance frontier", x=0.06, ha="left", fontsize=20, fontweight="bold", color=INK
    )

    _scatter_panel(
        left,
        stats,
        cost_index,
        label_y,
        "COST INDEX · CHEAPEST = 1x",
        "↑ BETTER · ← CHEAPER",
        log_x=True,
    )
    left.set_title("Score × Cost", loc="left", fontsize=14, fontweight="bold", color=INK, pad=18)
    left.set_ylabel("mean score (%)", fontsize=10, color=MUTED, fontfamily="monospace")

    _scatter_panel(
        right,
        stats,
        speed_index,
        label_y,
        "SPEED INDEX · FASTEST = 1x",
        "↑ BETTER · ← FASTER",
        log_x=False,
    )
    right.set_title("Score × Speed", loc="left", fontsize=14, fontweight="bold", color=INK, pad=18)

    scores = [s.mean_score for s in stats]
    pad = (max(scores) - min(scores)) * 0.18 + 1
    for ax in (left, right):
        ax.set_ylim(min(scores) - pad, max(scores) + pad)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.text(
        0.06,
        0.02,
        "Both axes are indexed to the best in class "
        "(cheapest = 1x, fastest = 1x). Speed uses median latency on passing tasks.",
        fontsize=10,
        color=MUTED,
        style="italic",
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.13, wspace=0.18)
    fig.savefig(out_path, dpi=150, facecolor=PAGE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Redraw the README figures from a leaderboard.")
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("results/frontier-2026-05-17/leaderboard.jsonl"),
        help="published leaderboard.jsonl to plot",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="directory to write the figures into",
    )
    args = parser.parse_args()

    if not args.leaderboard.is_file():
        parser.error(f"no leaderboard at {args.leaderboard}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.leaderboard)
    draw_alpha_ranking(stats, args.out_dir / "alpha-ranking.jpg")
    draw_performance_frontier(stats, args.out_dir / "performance-frontier.jpg")
    print(f"wrote {args.out_dir}/alpha-ranking.jpg and {args.out_dir}/performance-frontier.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
