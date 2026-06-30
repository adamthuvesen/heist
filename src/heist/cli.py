from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from heist.agents import (
    DEFAULT_AGENT_IDS,
    available_agents,
    resolve_agents,
)
from heist.config import (
    CONFIG_FILENAME,
    HeistConfig,
    LoadedConfig,
    load_config,
    parse_provider_jobs,
    render_starter_config,
)
from heist.export import export_eval_audit
from heist.history import (
    BaselineRegistry,
    ComparisonReport,
    HistoryError,
    RunSummary,
    compare,
    cross_run_table,
    load_all_runs,
    resolve_run_ref,
)
from heist.models import AgentSpec, RunManifest, TaskDefinition, TaskRunResult
from heist.paths import default_runs_dir, find_repo_root
from heist.progress import select_reporter
from heist.reporting import write_report
from heist.runner import (
    load_manifest,
    load_results,
    regrade_run,
    run_benchmark,
    validate_run_id,
)
from heist.tasks import list_suites, load_tasks, select_tasks

app = typer.Typer(
    help=(
        "HEIST — Hidden Evaluation of Integrated System Tasks. "
        "Local benchmark harness for CLI coding agents."
    ),
    no_args_is_help=True,
)
tasks_app = typer.Typer(help="Inspect benchmark tasks.", no_args_is_help=True)
agents_app = typer.Typer(help="Inspect the agent registry.", no_args_is_help=True)
suites_app = typer.Typer(help="Inspect task suites.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect HEIST configuration.", no_args_is_help=True)
export_app = typer.Typer(
    help="Export run artifacts for external audit tools.", no_args_is_help=True
)
runs_app = typer.Typer(help="Inspect and compare prior benchmark runs.", no_args_is_help=True)
baseline_app = typer.Typer(help="Manage named baseline tags.", no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")
app.add_typer(agents_app, name="agents")
app.add_typer(suites_app, name="suites")
app.add_typer(config_app, name="config")
app.add_typer(export_app, name="export")
app.add_typer(runs_app, name="runs")
runs_app.add_typer(baseline_app, name="baseline")

console = Console(highlight=False)


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_root(repo_root: Path | None) -> Path:
    return repo_root or find_repo_root()


def _load_config(root: Path) -> LoadedConfig:
    try:
        return load_config(root)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="config") from error


def _check_minimum(value: int, *, name: str, minimum: int, param_hint: str) -> int:
    if value < minimum:
        raise typer.BadParameter(f"{name} must be >= {minimum}, got {value}", param_hint=param_hint)
    return value


def _check_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    try:
        return validate_run_id(run_id)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--run-id") from error


def _error_text(error: Exception) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def _load_manifest(run_dir: Path, *, param_hint: str = "--run") -> RunManifest:
    try:
        return load_manifest(run_dir)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint=param_hint) from error


def _load_results(run_dir: Path, *, param_hint: str = "--run") -> list[TaskRunResult]:
    try:
        return load_results(run_dir)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint=param_hint) from error


def _load_tasks(
    *, suite: str, repo_root: Path, param_hint: str = "--suite"
) -> list[TaskDefinition]:
    try:
        return load_tasks(suite=suite, repo_root=repo_root)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint=param_hint) from error


def _select_tasks(
    *,
    suite: str,
    repo_root: Path,
    task_ids: list[str] | None = None,
    glob: str | None = None,
    category: str | None = None,
    param_hint: str = "--task",
) -> list[TaskDefinition]:
    try:
        return select_tasks(
            suite=suite,
            task_ids=task_ids,
            repo_root=repo_root,
            glob=glob,
            category=category,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint=param_hint) from error


def _available_agents(
    *,
    agent_file: Path | None,
    extra_files: list[Path],
    param_hint: str = "--agent-file",
) -> dict[str, AgentSpec]:
    try:
        return available_agents(agent_file=agent_file, extra_files=extra_files)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint=param_hint) from error


def _resolve_agents(
    *,
    agent_ids: list[str] | None,
    agent_file: Path | None,
    extra_files: list[Path],
    all_agents: bool = False,
    providers: list[str] | None = None,
    exclude: list[str] | None = None,
    default_set: list[str] | None = None,
) -> list[AgentSpec]:
    try:
        return resolve_agents(
            agent_ids=agent_ids,
            agent_file=agent_file,
            extra_files=extra_files,
            all_agents=all_agents,
            providers=providers,
            exclude=exclude,
            default_set=default_set,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise typer.BadParameter(_error_text(error), param_hint="--agent") from error


def _agent_file_paths(loaded: LoadedConfig, root: Path) -> list[Path]:
    paths: list[Path] = []
    for entry in loaded.agents.files:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path
        paths.append(path)
    return paths


@app.command()
def run(
    suite: Annotated[
        str | None, typer.Option(help="Task suite to run (default from config).")
    ] = None,
    agent: Annotated[
        list[str] | None,
        typer.Option("--agent", help="Agent id to run. Repeat for multiple agents."),
    ] = None,
    all_agents: Annotated[
        bool,
        typer.Option("--all-agents", help="Run every agent in the registry."),
    ] = False,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            "--provider",
            help="Run every agent whose provider matches. Repeat for multiple.",
        ),
    ] = None,
    exclude_agent: Annotated[
        list[str] | None,
        typer.Option("--exclude-agent", help="Drop a specific agent from the selection."),
    ] = None,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Task id to run. Repeat for a subset."),
    ] = None,
    task_glob: Annotated[
        str | None,
        typer.Option("--task-glob", help="fnmatch pattern over task ids (e.g. 'ledger-*')."),
    ] = None,
    task_category: Annotated[
        str | None,
        typer.Option("--task-category", help="Filter selection by TaskSpec.category."),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(help="Per-task timeout in seconds (default from config)."),
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option(help="Maximum concurrent (agent, task) jobs (default from config)."),
    ] = None,
    provider_jobs: Annotated[
        str | None,
        typer.Option(
            "--provider-jobs",
            help="Per-provider concurrency caps, e.g. 'claude=3,cursor=4,codex=2'.",
        ),
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Optional stable run id.")] = None,
    agent_file: Annotated[
        Path | None,
        typer.Option(help="YAML file with additional agent definitions."),
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Directory for run artifacts.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the (agent, task) matrix and exit 0."),
    ] = False,
    retry: Annotated[
        int,
        typer.Option(help="Retry agent invocation on non-grader exceptions."),
    ] = 0,
    progress: Annotated[
        bool | None,
        typer.Option(
            "--progress/--no-progress",
            help="Enable/disable live progress UI (auto-off when not a TTY).",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Disable progress UI; print only final paths."),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase log verbosity (-v, -vv)."),
    ] = 0,
    exit_on_failure: Annotated[
        bool | None,
        typer.Option(
            "--exit-on-failure/--no-exit-on-failure",
            help="Exit non-zero if any task errors (1) or fails grading (2).",
        ),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Cancel pending jobs after the first error."),
    ] = False,
    sandbox: Annotated[
        bool | None,
        typer.Option(
            "--sandbox/--no-sandbox",
            help="Wrap the agent in a sandbox-exec profile that denies reads of "
            "tasks/ (hidden graders + references). macOS only.",
        ),
    ] = None,
) -> None:
    """Execute the benchmark."""
    _configure_logging(verbose)
    root = _resolve_root(repo_root)
    cfg = _load_config(root)

    effective_suite = suite or cfg.defaults.suite
    effective_jobs = _check_minimum(
        jobs if jobs is not None else cfg.defaults.jobs,
        name="jobs",
        minimum=1,
        param_hint="--jobs",
    )
    effective_timeout = _check_minimum(
        timeout if timeout is not None else cfg.defaults.timeout_s,
        name="timeout",
        minimum=1,
        param_hint="--timeout",
    )
    run_id = _check_run_id(run_id)
    retry = _check_minimum(retry, name="retry", minimum=0, param_hint="--retry")
    if quiet:
        effective_progress = False
    elif progress is not None:
        effective_progress = progress
    else:
        effective_progress = cfg.defaults.progress
    effective_exit_on_failure = (
        exit_on_failure if exit_on_failure is not None else cfg.defaults.exit_on_failure
    )
    effective_sandbox = sandbox if sandbox is not None else cfg.defaults.sandbox
    if effective_sandbox and sys.platform != "darwin":
        raise typer.BadParameter(
            "--sandbox requires macOS (sandbox-exec). The cheat-detector runs "
            "regardless; use container isolation on this platform.",
            param_hint="--sandbox",
        )
    runs_dir = output_dir or (
        root / cfg.defaults.output_dir if cfg.defaults.output_dir else default_runs_dir(root)
    )

    selected_agents = _resolve_agents(
        agent_ids=agent,
        agent_file=agent_file,
        extra_files=_agent_file_paths(cfg, root),
        all_agents=all_agents,
        providers=provider,
        exclude=exclude_agent,
        default_set=cfg.selection.default_agents or None,
    )
    selected_tasks = _select_tasks(
        suite=effective_suite,
        task_ids=task,
        repo_root=root,
        glob=task_glob,
        category=task_category,
    )

    try:
        provider_caps = parse_provider_jobs(provider_jobs)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--provider-jobs") from error
    if not provider_caps and cfg.providers:
        provider_caps = dict(cfg.providers)

    if dry_run:
        _print_dry_run(
            agents=selected_agents,
            tasks=selected_tasks,
            jobs=effective_jobs,
            provider_caps=provider_caps,
            suite=effective_suite,
            timeout_s=effective_timeout,
        )
        return

    job_pairs = [(a, t) for a in selected_agents for t in selected_tasks]
    reporter = select_reporter(
        jobs=job_pairs,
        jobs_cap=effective_jobs,
        progress=effective_progress,
    )

    with reporter:
        manifest, results = run_benchmark(
            repo_root=root,
            suite=effective_suite,
            agents=selected_agents,
            tasks=selected_tasks,
            runs_dir=runs_dir,
            timeout_s=effective_timeout,
            run_id=run_id,
            jobs=effective_jobs,
            provider_jobs=provider_caps,
            reporter=reporter,
            retry=retry,
            fail_fast=fail_fast,
            sandbox=effective_sandbox,
        )

    report_path = write_report(Path(manifest.run_dir), results)
    console.print(f"Run written to [cyan]{manifest.run_dir}[/cyan]")
    console.print(f"Report written to [cyan]{report_path}[/cyan]")

    if effective_exit_on_failure:
        raise typer.Exit(code=_exit_code_for(results))


def _exit_code_for(results: list[TaskRunResult]) -> int:
    if any(result.outcome_status == "errored" for result in results):
        return 1
    if any(result.success is False for result in results):
        return 2
    return 0


def _print_dry_run(
    *,
    agents: list[AgentSpec],
    tasks: list[TaskDefinition],
    jobs: int,
    provider_caps: dict[str, int],
    suite: str,
    timeout_s: int,
) -> None:
    table = Table(
        title=(
            f"dry-run · suite={suite} · jobs={jobs} · timeout={timeout_s}s · "
            f"{len(agents) * len(tasks)} pairs"
        ),
        expand=False,
    )
    table.add_column("Agent")
    table.add_column("Task")
    table.add_column("Category")
    table.add_column("Provider")
    table.add_column("Provider cap", justify="right")
    default_caps = {agent.provider: jobs for agent in agents}
    default_caps.update(provider_caps)
    for agent in agents:
        cap = default_caps.get(agent.provider, jobs)
        for task in tasks:
            table.add_row(
                agent.id,
                task.id,
                task.spec.category,
                agent.provider,
                str(min(cap, jobs)),
            )
    console.print(table)


@tasks_app.command("list")
def tasks_list(
    suite: Annotated[str | None, typer.Option(help="Suite to list.")] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """List tasks in a suite."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    suite_name = suite or cfg.defaults.suite
    tasks = _load_tasks(suite=suite_name, repo_root=root)

    table = Table(title=f"Suite: {suite_name}", expand=False)
    table.add_column("Task id")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Difficulty")
    for task in tasks:
        table.add_row(task.id, task.spec.title, task.spec.category, task.spec.difficulty)
    console.print(table)


@tasks_app.command("show")
def tasks_show(
    task_id: Annotated[str, typer.Argument(help="Task id to inspect.")],
    suite: Annotated[str | None, typer.Option(help="Suite that contains the task.")] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Show full details for a single task."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    suite_name = suite or cfg.defaults.suite
    [task] = _select_tasks(
        suite=suite_name,
        task_ids=[task_id],
        repo_root=root,
        param_hint="task_id",
    )

    console.print(f"[bold]{task.id}[/bold] — {task.spec.title}")
    console.print(f"category: {task.spec.category}   difficulty: {task.spec.difficulty}")
    if task.spec.timeout_s:
        console.print(f"timeout_s: {task.spec.timeout_s}")
    console.print(f"visible test: [dim]{' '.join(task.spec.visible_test_command)}[/dim]")
    console.print()
    console.print("[bold]Prompt[/bold]")
    console.print(task.spec.prompt.strip())
    console.print()

    tree = Tree(f"[bold]Workspace[/bold] {task.workspace_path}")
    for path in sorted(task.workspace_path.iterdir()):
        tree.add(path.name + ("/" if path.is_dir() else ""))
    console.print(tree)


@agents_app.command("list")
def agents_list(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Only show agents for this provider."),
    ] = None,
    agent_file: Annotated[
        Path | None,
        typer.Option(help="Merge agents from this YAML file before listing."),
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """List known agents."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    agents = _available_agents(
        agent_file=agent_file,
        extra_files=_agent_file_paths(cfg, root),
    )

    table = Table(title=f"{len(agents)} known agents", expand=False)
    table.add_column("Agent id")
    table.add_column("Label")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Default", justify="center")

    defaults = set(DEFAULT_AGENT_IDS)
    for agent_id, spec in sorted(agents.items()):
        if provider and spec.provider.lower() != provider.lower():
            continue
        table.add_row(
            agent_id,
            spec.label,
            spec.provider,
            spec.model_id,
            "yes" if agent_id in defaults else "",
        )
    console.print(table)


@agents_app.command("show")
def agents_show(
    agent_id: Annotated[str, typer.Argument(help="Agent id to inspect.")],
    agent_file: Annotated[
        Path | None, typer.Option(help="Merge agents from this YAML file.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Show full spec for a single agent."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    agents = _available_agents(
        agent_file=agent_file,
        extra_files=_agent_file_paths(cfg, root),
    )
    if agent_id not in agents:
        known = ", ".join(sorted(agents))
        raise typer.BadParameter(f"Unknown agent {agent_id!r}. Known: {known}")
    body = json.dumps(agents[agent_id].model_dump(), indent=2)
    console.print(Syntax(body, "json", theme="ansi_dark"))


@suites_app.command("list")
def suites_list(
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """List task suites."""
    root = _resolve_root(repo_root)
    suites = list_suites(root)
    table = Table(title="Suites", expand=False)
    table.add_column("Suite")
    table.add_column("Tasks", justify="right")
    for name in suites:
        tasks = _load_tasks(suite=name, repo_root=root)
        table.add_row(name, str(len(tasks)))
    console.print(table)


@config_app.command("show")
def config_show(
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Print effective merged config."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    body = _render_effective_config(cfg.config)
    console.print(Syntax(body, "toml", theme="ansi_dark"))


@config_app.command("path")
def config_path(
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Show which config sources were read."""
    root = _resolve_root(repo_root)
    cfg = _load_config(root)
    if not cfg.sources:
        console.print("[dim]No config files found; using built-in defaults.[/dim]")
        return
    for source in cfg.sources:
        console.print(str(source))


@config_app.command("init")
def config_init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing heist.toml.")
    ] = False,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Write a starter heist.toml."""
    root = _resolve_root(repo_root)
    target = root / CONFIG_FILENAME
    if target.exists() and not force:
        raise typer.BadParameter(
            f"{target} already exists. Pass --force to overwrite.",
            param_hint="--force",
        )
    target.write_text(render_starter_config())
    console.print(f"Wrote starter config to [cyan]{target}[/cyan]")


@app.command()
def grade(
    run: Annotated[Path, typer.Option("--run", help="Run directory to re-grade.")],
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Re-grade an existing run."""
    root = _resolve_root(repo_root)
    tasks = _select_tasks(suite=_load_manifest(run).suite, repo_root=root)
    try:
        results = regrade_run(run, tasks)
    except ValueError as error:
        raise typer.BadParameter(_error_text(error), param_hint="--run") from error
    console.print(f"Regraded {len(results)} rows into [cyan]{run / 'regrade-results.jsonl'}[/cyan]")


@app.command()
def report(
    run: Annotated[Path, typer.Option("--run", help="Run directory to report.")],
    compare_baseline: Annotated[
        str | None,
        typer.Option(
            "--compare-baseline",
            help=(
                "Baseline run reference (tag, run id, or 'latest'/'previous') to "
                "render delta columns against in the HTML report."
            ),
        ),
    ] = None,
    runs_dir: Annotated[
        Path | None,
        typer.Option(
            "--runs-dir",
            help=(
                "Override runs/ directory (used to resolve --compare-baseline). "
                "Defaults to the parent of --run."
            ),
        ),
    ] = None,
) -> None:
    """Render the markdown + HTML report for an existing run."""
    results = _load_results(run)
    baseline_comparison = None
    if compare_baseline is not None:
        runs_root = runs_dir if runs_dir is not None else run.resolve().parent
        try:
            baseline_run, banner = resolve_run_ref(runs_root, compare_baseline)
        except HistoryError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from error
        if banner:
            console.print(f"[yellow]⚠ {banner}[/yellow]")
        baseline_comparison = compare(runs_root, baseline_run, run.name)
    manifest = _load_manifest(run)
    replay_source_run_id = manifest.source_run_id if manifest.kind == "replay" else None
    path = write_report(
        run,
        results,
        baseline_comparison=baseline_comparison,
        replay_source_run_id=replay_source_run_id,
    )
    sys.stdout.write(path.read_text())


@export_app.command("eval-audit")
def eval_audit(
    run: Annotated[Path, typer.Option("--run", help="Run directory to export.")],
) -> None:
    """Export run results to eval-audit parquet."""
    results = _load_results(run)
    path = export_eval_audit(run, results)
    console.print(f"eval-audit parquet written to [cyan]{path}[/cyan]")


# ---------------------------------------------------------------------------
# `heist runs` — cross-run analysis
# ---------------------------------------------------------------------------


def _resolve_runs_dir(runs_dir: Path | None, repo_root: Path | None) -> Path:
    if runs_dir is not None:
        return runs_dir
    return default_runs_dir(_resolve_root(repo_root))


def _short_sha(value: str | None) -> str:
    if not value:
        return "—"
    return value[:12]


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _format_cost_cell(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _format_latency(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


def _format_delta(value: float | None, *, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "score":
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value) * 100:.1f}pp"
    if kind == "cost":
        sign = "+" if value >= 0 else "−"
        return f"{sign}${abs(value):.4f}"
    if kind == "latency":
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value):.1f}s"
    return f"{value:+.3f}"


def _report_run_summary(summary: RunSummary) -> str:
    return (
        f"{summary.run_id}  ({summary.kind})  "
        f"suite={summary.suite}  "
        f"sha={_short_sha(summary.harness_git_sha)}"
    )


@runs_app.command("list")
def runs_list(
    runs_dir: Annotated[
        Path | None,
        typer.Option(
            "--runs-dir",
            help="Override the `runs/` directory (default: <repo_root>/runs).",
        ),
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
    include_corrupt: Annotated[
        bool,
        typer.Option(
            "--include-corrupt",
            help="Also list run dirs with unreadable manifests.",
        ),
    ] = False,
) -> None:
    """List all runs in `runs/` sorted by most recent first."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)
    summaries, corrupt = load_all_runs(runs_root, include_corrupt=include_corrupt)
    if not summaries and not corrupt:
        console.print(
            f"No runs found under [cyan]{runs_root}[/cyan]. "
            "Generate one with [bold]heist run[/bold]."
        )
        return

    table = Table(title=f"{len(summaries)} runs", expand=False)
    table.add_column("Run id")
    table.add_column("Kind")
    table.add_column("Suite")
    table.add_column("Agents", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Mean score", justify="right")
    table.add_column("Total cost", justify="right")
    table.add_column("Created at")
    table.add_column("SHA")
    table.add_column("Tags")
    for summary in summaries:
        table.add_row(
            summary.run_id,
            summary.kind,
            summary.suite,
            str(len(summary.agent_ids)),
            str(len(summary.task_ids)),
            _format_score(summary.mean_score),
            _format_cost_cell(summary.total_cost_usd),
            summary.created_at.strftime("%Y-%m-%d %H:%M"),
            _short_sha(summary.harness_git_sha),
            ", ".join(summary.tags) if summary.tags else "—",
        )
    console.print(table)

    if corrupt:
        console.print(f"[yellow]{len(corrupt)} run dir(s) had unreadable manifests:[/yellow]")
        for entry in corrupt:
            console.print(f"  • {entry.run_id}: {entry.reason}")


def _render_compare_report(report: ComparisonReport) -> None:
    console.print(f"[bold]A[/bold]  {_report_run_summary(report.run_a)}")
    console.print(f"[bold]B[/bold]  {_report_run_summary(report.run_b)}")
    if report.harness_drift:
        console.print(f"[yellow]⚠ {report.harness_drift}[/yellow]")
    if report.run_a.run_id == report.run_b.run_id:
        console.print("[dim]Runs are identical — all deltas are zero.[/dim]")

    if report.agents_only_in_a or report.agents_only_in_b:
        if report.agents_only_in_a:
            console.print(
                f"[yellow]Agents only in A:[/yellow] {', '.join(report.agents_only_in_a)}"
            )
        if report.agents_only_in_b:
            console.print(
                f"[yellow]Agents only in B:[/yellow] {', '.join(report.agents_only_in_b)}"
            )

    if not report.rows:
        console.print("[dim]No shared (agent, task) pairs between the two runs.[/dim]")
    else:
        by_agent: dict[str, list] = {}
        for row in report.rows:
            by_agent.setdefault(row.agent_id, []).append(row)
        for agent_id in sorted(by_agent):
            agent_rows = by_agent[agent_id]
            table = Table(title=f"Agent: {agent_id}", expand=False)
            table.add_column("Task")
            table.add_column("Score A", justify="right")
            table.add_column("Score B", justify="right")
            table.add_column("Δ score", justify="right")
            table.add_column("Δ latency", justify="right")
            table.add_column("Δ cost", justify="right")
            table.add_column("Note")
            for row in agent_rows:
                note = ""
                if row.regression == "pass_to_fail":
                    note = "[red]pass → fail[/red]"
                elif row.regression == "score_drop":
                    note = "[red]score drop[/red]"
                elif row.outcome_status_a != row.outcome_status_b:
                    note = f"{row.outcome_status_a} → {row.outcome_status_b}"
                table.add_row(
                    row.task_id,
                    _format_score(row.score_a),
                    _format_score(row.score_b),
                    _format_delta(row.delta_score, kind="score"),
                    _format_delta(row.delta_latency_s, kind="latency"),
                    _format_delta(row.delta_cost_usd, kind="cost"),
                    note,
                )
            console.print(table)

    if report.tasks_only_in_a or report.tasks_only_in_b:
        if report.tasks_only_in_a:
            console.print(f"[yellow]Tasks only in A:[/yellow] {', '.join(report.tasks_only_in_a)}")
        if report.tasks_only_in_b:
            console.print(f"[yellow]Tasks only in B:[/yellow] {', '.join(report.tasks_only_in_b)}")


@runs_app.command("compare")
def runs_compare(
    a: Annotated[
        str | None,
        typer.Argument(
            help=(
                "First run reference (id, baseline tag, 'latest', 'previous'). "
                "Omit when using --baseline."
            ),
        ),
    ] = None,
    b: Annotated[
        str | None,
        typer.Argument(
            help="Second run reference. Omit when using --baseline.",
        ),
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option(
            "--baseline",
            help=(
                "Compare a single run against a baseline tag — equivalent to "
                "`compare <baseline> <run>`. Pass only the run as the first "
                "positional argument."
            ),
        ),
    ] = None,
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
    literal: Annotated[
        bool,
        typer.Option(
            "--literal",
            help="Treat both refs as literal run ids (bypass tag resolution).",
        ),
    ] = False,
) -> None:
    """Compare two runs side by side, surfacing score / cost / latency deltas."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)

    try:
        if baseline is not None:
            if b is not None:
                console.print("[red]Pass exactly one positional run when using --baseline.[/red]")
                raise typer.Exit(code=2)
            if a is None:
                console.print("[red]Provide the run to compare against --baseline.[/red]")
                raise typer.Exit(code=2)
            ref_a = baseline
            ref_b = a
        else:
            if a is None or b is None:
                console.print("[red]Pass two run references, or one ref plus --baseline.[/red]")
                raise typer.Exit(code=2)
            ref_a = a
            ref_b = b

        summaries, _ = load_all_runs(runs_root)
        registry = BaselineRegistry.load(runs_root)
        run_a, banner_a = resolve_run_ref(
            runs_root, ref_a, literal=literal, summaries=summaries, baselines=registry
        )
        run_b, banner_b = resolve_run_ref(
            runs_root, ref_b, literal=literal, summaries=summaries, baselines=registry
        )
        for banner in (banner_a, banner_b):
            if banner:
                console.print(f"[yellow]⚠ {banner}[/yellow]")
        report = compare(runs_root, run_a, run_b)
    except HistoryError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if baseline is not None:
        console.print(f"[dim]baseline: {baseline} → {run_a}[/dim]")
    _render_compare_report(report)


@runs_app.command("history")
def runs_history(
    agent: Annotated[str, typer.Option("--agent", help="Agent id to filter on.")],
    task: Annotated[str, typer.Option("--task", help="Task id to filter on.")],
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Show every run in which the (agent, task) pair appeared, chronologically."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)
    summaries, _ = load_all_runs(runs_root)
    known_agents = {aid for s in summaries for aid in s.agent_ids}
    known_tasks = {tid for s in summaries for tid in s.task_ids}
    if known_agents and agent not in known_agents:
        console.print(
            f"[red]Unknown agent id {agent!r}. Known agents: "
            f"{', '.join(sorted(known_agents))}[/red]"
        )
        raise typer.Exit(code=1)
    if known_tasks and task not in known_tasks:
        console.print(
            f"[red]Unknown task id {task!r}. Known tasks: {', '.join(sorted(known_tasks))}[/red]"
        )
        raise typer.Exit(code=1)

    rows = cross_run_table(runs_root, agent_id=agent, task_id=task)
    if not rows:
        console.print(f"No history for agent [cyan]{agent}[/cyan] on task [cyan]{task}[/cyan].")
        return

    table = Table(title=f"History — agent={agent} task={task}", expand=False)
    table.add_column("Run id")
    table.add_column("Created at")
    table.add_column("Score", justify="right")
    table.add_column("Outcome")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("SHA")
    for row in rows:
        table.add_row(
            str(row["run_id"]),
            row["created_at"].strftime("%Y-%m-%d %H:%M"),  # type: ignore[union-attr]
            _format_score(float(row["score"])),
            str(row["outcome_status"]),
            _format_latency(float(row["latency_s"]) if row["latency_s"] is not None else None),
            _format_cost_cell(float(row["cost_usd"]) if row["cost_usd"] is not None else None),
            _short_sha(str(row["harness_git_sha"]) if row["harness_git_sha"] else None),
        )
    console.print(table)

    scores = [float(row["score"]) for row in rows]
    console.print(
        f"[dim]n={len(scores)}  min={min(scores) * 100:.1f}%  "
        f"max={max(scores) * 100:.1f}%  mean={sum(scores) / len(scores) * 100:.1f}%[/dim]"
    )


def _replay_agents_from_source(
    source_results: list[TaskRunResult],
) -> list[AgentSpec]:
    """Build minimal AgentSpec instances from a source run's result rows.

    Replay never invokes the agent CLI, so `command` is a placeholder and
    `provider` is "replay" — these never reach a subprocess. Preserving
    `id`, `label`, and `model_id` keeps downstream rows and reports
    consistent with the source.
    """
    seen: dict[str, TaskRunResult] = {}
    for row in source_results:
        seen.setdefault(row.agent_id, row)
    return [
        AgentSpec(
            id=row.agent_id,
            label=row.agent_label,
            provider="replay",
            model_id=row.model_id,
            command=["true"],
        )
        for row in seen.values()
    ]


def _select_replay_ids(
    *,
    source_run_id: str,
    kind: str,
    available: list[str],
    requested: list[str] | None,
    exclude: list[str] | None = None,
) -> list[str]:
    if requested:
        unknown = [value for value in requested if value not in available]
        if unknown:
            raise ValueError(
                f"Source run {source_run_id} has no rows for {kind}(s): "
                f"{', '.join(unknown)}. Known {kind}s: {', '.join(available)}"
            )
        requested_set = set(requested)
        selected = [value for value in available if value in requested_set]
    else:
        selected = list(available)

    if exclude:
        excluded = set(exclude)
        selected = [value for value in selected if value not in excluded]
    return selected


@runs_app.command("replay")
def runs_replay(
    source: Annotated[
        str,
        typer.Argument(
            help=(
                "Source run reference (literal id, baseline tag, 'latest', "
                "or 'previous'). Replay reconstructs each (agent, task) "
                "execution from this run's captured artefacts."
            ),
        ),
    ],
    agent: Annotated[
        list[str] | None,
        typer.Option("--agent", help="Only replay these agent ids (repeatable)."),
    ] = None,
    exclude_agent: Annotated[
        list[str] | None,
        typer.Option("--exclude-agent", help="Drop agents from the replay selection."),
    ] = None,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Only replay these task ids (repeatable)."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Optional stable run id for the replay output."),
    ] = None,
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
    literal: Annotated[
        bool,
        typer.Option(
            "--literal",
            help="Treat <source> as a literal run id (bypass tag resolution).",
        ),
    ] = False,
) -> None:
    """Re-grade and re-report a prior run without invoking any agent CLI.

    Replay uses captured stdout/stderr and workspace state from the source
    run to reproduce per-task outcomes. The grader, cost pipeline, and
    reports run live, so this is the right surface for testing harness or
    grader changes against fixed agent behaviour.

    Note: costs are recomputed via the current pricing table. A pricing
    change since the source run will shift `reconstructed_per_task_cost_usd`
    in the replay even though `usage` and `reported_session_cost_usd`
    match the source verbatim.
    """
    root = _resolve_root(repo_root)
    runs_root = _resolve_runs_dir(runs_dir, repo_root)

    # Late imports keep replay's runtime dependencies off the hot CLI path.
    from heist.replay import (
        ReplayOfReplayError,
        SnapshotExecutor,
    )

    try:
        source_run_id, banner = resolve_run_ref(runs_root, source, literal=literal)
    except HistoryError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    if banner:
        console.print(f"[yellow]⚠ {banner}[/yellow]")

    source_dir = runs_root / source_run_id
    try:
        executor = SnapshotExecutor(source_dir)
    except ReplayOfReplayError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    source_manifest = _load_manifest(source_dir)
    source_results = _load_results(source_dir)

    selectable_agents = sorted({row.agent_id for row in source_results})
    selectable_tasks = sorted({row.task_id for row in source_results})

    try:
        chosen_agents = _select_replay_ids(
            source_run_id=source_run_id,
            kind="agent",
            available=selectable_agents,
            requested=agent,
            exclude=exclude_agent,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    if not chosen_agents:
        console.print("[red]No agents selected for replay after filters applied.[/red]")
        raise typer.Exit(code=1)

    try:
        chosen_tasks = _select_replay_ids(
            source_run_id=source_run_id,
            kind="task",
            available=selectable_tasks,
            requested=task,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    filtered_rows = [
        row
        for row in source_results
        if row.agent_id in chosen_agents and row.task_id in chosen_tasks
    ]
    replay_agents = [a for a in _replay_agents_from_source(filtered_rows) if a.id in chosen_agents]
    tasks_for_run = _select_tasks(
        suite=source_manifest.suite,
        task_ids=chosen_tasks,
        repo_root=root,
    )

    replay_run_id = run_id or f"replay-of-{source_run_id}"
    console.print(
        f"[bold]Replay[/bold]  source={source_run_id} → new run "
        f"[cyan]{replay_run_id}[/cyan]  ({len(replay_agents)} agents × "
        f"{len(tasks_for_run)} tasks)"
    )
    console.print(
        "[dim]This run does not measure agents. It re-grades and re-reports "
        "the captured outputs of the source run.[/dim]"
    )

    manifest, results = run_benchmark(
        repo_root=root,
        suite=source_manifest.suite,
        agents=replay_agents,
        tasks=tasks_for_run,
        runs_dir=runs_root,
        timeout_s=1,  # unused: SnapshotExecutor never blocks on a subprocess.
        run_id=replay_run_id,
        executor=executor,
        kind="replay",
        source_run_id=source_run_id,
    )
    write_report(Path(manifest.run_dir), results, replay_source_run_id=source_run_id)
    console.print(f"Replay written to [cyan]{manifest.run_dir}[/cyan] ({len(results)} rows)")


@baseline_app.command("set")
def baseline_set(
    run: Annotated[str, typer.Argument(help="Run id (literal) to tag as baseline.")],
    tag: Annotated[str, typer.Argument(help="Tag name to assign.")],
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Tag a run as a baseline."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)
    registry = BaselineRegistry.load(runs_root)
    try:
        previous = registry.set(runs_root, run, tag)
    except HistoryError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    if previous is not None and previous != run:
        console.print(f"Reassigned baseline [bold]{tag}[/bold]: {previous} → {run}")
    else:
        console.print(f"Set baseline [bold]{tag}[/bold] → {run}")


@baseline_app.command("unset")
def baseline_unset(
    tag: Annotated[str, typer.Argument(help="Tag name to remove.")],
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """Remove a baseline tag."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)
    registry = BaselineRegistry.load(runs_root)
    try:
        removed = registry.unset(runs_root, tag)
    except HistoryError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    console.print(f"Removed baseline [bold]{tag}[/bold] (was → {removed}).")


@baseline_app.command("list")
def baseline_list(
    runs_dir: Annotated[
        Path | None, typer.Option("--runs-dir", help="Override runs/ directory.")
    ] = None,
    repo_root: Annotated[Path | None, typer.Option(help="HEIST repo root.")] = None,
) -> None:
    """List baseline tags."""
    runs_root = _resolve_runs_dir(runs_dir, repo_root)
    registry = BaselineRegistry.load(runs_root)
    entries = registry.list()
    if not entries:
        console.print("No baseline tags configured.")
        return
    table = Table(title="Baseline tags", expand=False)
    table.add_column("Tag")
    table.add_column("Run id")
    for tag, run_id in sorted(entries.items()):
        table.add_row(tag, run_id)
    console.print(table)


def _render_effective_config(cfg: HeistConfig) -> str:
    lines: list[str] = []
    defaults = cfg.defaults.model_dump()
    lines.append("[defaults]")
    for key, value in defaults.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if cfg.providers:
        lines.append("")
        lines.append("[providers]")
        for name, cap in cfg.providers.items():
            lines.append(f"{name} = {cap}")
    if cfg.selection.default_agents:
        lines.append("")
        lines.append("[selection]")
        joined = ", ".join(_toml_value(item) for item in cfg.selection.default_agents)
        lines.append(f"default_agents = [{joined}]")
    if cfg.agents.files:
        lines.append("")
        lines.append("[agents]")
        joined = ", ".join(_toml_value(item) for item in cfg.agents.files)
        lines.append(f"files = [{joined}]")
    return "\n".join(lines) + "\n"


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))
