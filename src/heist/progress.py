from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from heist.models import AgentSpec, TaskDefinition, TaskRunResult


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:d}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return " n/a"
    return f"${value:.4f}"


class ProgressReporter(Protocol):
    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None: ...
    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None: ...
    def __enter__(self) -> ProgressReporter: ...
    def __exit__(self, *args: object) -> None: ...


@dataclass
class _JobState:
    agent: AgentSpec
    task: TaskDefinition
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "queued"
    score: float | None = None
    cost: float | None = None
    error: str | None = None
    timed_out: bool = False


class NullReporter:
    """No-op reporter for use when nothing should be printed."""

    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None:
        return None

    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None:
        return None

    def __enter__(self) -> NullReporter:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@dataclass
class PlainReporter:
    """One line per job state transition. Log-friendly, no TTY required."""

    console: Console = field(default_factory=lambda: Console(stderr=False, highlight=False))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __enter__(self) -> PlainReporter:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None:
        with self._lock:
            self.console.print(f"[cyan]start[/cyan]  agent={agent.id} task={task.id}")

    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None:
        with self._lock:
            if result.outcome_status == "errored":
                tag = "[red]error[/red]"
                detail = "timed out" if result.timed_out else (result.error or "errored")
                self.console.print(
                    f"{tag}  agent={agent.id} task={task.id} "
                    f"time={_fmt_duration(result.latency_s or 0.0)} "
                    f"cost={_fmt_cost(result.cost_usd)} ({detail})"
                )
                return
            tag = "[green]pass[/green] " if result.success else "[yellow]fail[/yellow] "
            self.console.print(
                f"{tag} agent={agent.id} task={task.id} "
                f"score={result.score:.2f} "
                f"time={_fmt_duration(result.latency_s or 0.0)} "
                f"cost={_fmt_cost(result.cost_usd)}"
            )


class RichLiveReporter:
    """Live rich table for parallel runs in a TTY."""

    def __init__(self, jobs: list[tuple[AgentSpec, TaskDefinition]], *, jobs_cap: int) -> None:
        self._jobs_cap = jobs_cap
        self._states: dict[tuple[str, str], _JobState] = {
            (agent.id, task.id): _JobState(agent=agent, task=task) for agent, task in jobs
        }
        self._order: list[tuple[str, str]] = [(agent.id, task.id) for agent, task in jobs]
        self._console = Console(highlight=False)
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._live: Live | None = None

    def __enter__(self) -> RichLiveReporter:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)
            self._live.__exit__(*args)
            self._live = None

    def on_start(self, agent: AgentSpec, task: TaskDefinition) -> None:
        with self._lock:
            state = self._states[(agent.id, task.id)]
            state.status = "running"
            state.started_at = time.monotonic()
        self._refresh()

    def on_finish(self, agent: AgentSpec, task: TaskDefinition, result: TaskRunResult) -> None:
        with self._lock:
            state = self._states[(agent.id, task.id)]
            state.finished_at = time.monotonic()
            state.score = result.score
            state.cost = result.cost_usd
            state.timed_out = result.timed_out
            if result.outcome_status == "errored":
                state.status = "timeout" if result.timed_out else "error"
                state.error = result.error
            elif result.success:
                state.status = "pass"
            else:
                state.status = "fail"
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=False)

    def _render(self) -> Table:
        # Snapshot under the lock so a worker thread mutating a _JobState mid-render
        # can't produce a torn row (e.g. status updated but score not yet). Shallow
        # copies are enough — every mutated field is a scalar.
        with self._lock:
            states = [replace(self._states[key]) for key in self._order]

        total = len(states)
        done = sum(1 for s in states if s.finished_at is not None)
        failed = sum(1 for s in states if s.status in {"fail", "error", "timeout"})
        running = sum(1 for s in states if s.status == "running")
        elapsed = time.monotonic() - self._started_at
        eta = self._estimate_eta(done, total, elapsed)

        title = (
            f"{done}/{total} done · {failed} failed · {running} running · "
            f"jobs={self._jobs_cap} · elapsed={_fmt_duration(elapsed)} · ETA {eta}"
        )

        table = Table(title=title, expand=True, show_header=True, header_style="bold")
        table.add_column("Agent", overflow="fold")
        table.add_column("Task", overflow="fold")
        table.add_column("Status", width=10)
        table.add_column("Elapsed", justify="right", width=8)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Cost", justify="right", width=10)

        now = time.monotonic()
        for state in states:
            elapsed_text = self._elapsed_text(state, now)
            table.add_row(
                state.agent.id,
                state.task.id,
                self._status_cell(state),
                elapsed_text,
                f"{state.score:.2f}" if state.score is not None else "  -",
                _fmt_cost(state.cost),
            )
        return table

    @staticmethod
    def _elapsed_text(state: _JobState, now: float) -> str:
        if state.started_at is None:
            return " -"
        end = state.finished_at or now
        return _fmt_duration(end - state.started_at)

    @staticmethod
    def _status_cell(state: _JobState) -> Text:
        text = state.status
        style = {
            "queued": "dim",
            "running": "cyan",
            "pass": "green",
            "fail": "yellow",
            "error": "red",
            "timeout": "red",
        }.get(text, "white")
        return Text(text, style=style)

    @staticmethod
    def _estimate_eta(done: int, total: int, elapsed: float) -> str:
        if done == 0 or done >= total:
            return "--"
        per_job = elapsed / done
        remaining = max(0.0, per_job * (total - done))
        return _fmt_duration(remaining)


def select_reporter(
    *,
    jobs: list[tuple[AgentSpec, TaskDefinition]],
    jobs_cap: int,
    progress: bool,
    force_plain: bool = False,
) -> ProgressReporter:
    """Pick the right reporter for the current invocation."""
    if not progress:
        return NullReporter()
    if force_plain or not sys.stdout.isatty():
        return PlainReporter()
    return RichLiveReporter(jobs, jobs_cap=jobs_cap)
