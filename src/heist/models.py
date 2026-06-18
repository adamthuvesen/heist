from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OutcomeStatus = Literal["graded", "errored"]
RunStatus = Literal["in_progress", "completed", "aborted"]
CostProvenance = Literal[
    "reconciled", "partial", "extrapolated", "as_reported_only", "cost_not_available"
]
CostSource = Literal["reported", "reconstructed", "estimated", "unavailable"]


class CheckResult(BaseModel):
    name: str
    passed: bool
    message: str = ""


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    checks: list[CheckResult]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # `id` becomes a directory name under runs/<run_id>/workspaces/<agent>/<id>.
    # The pattern forbids slashes and `..` so a malformed task.yaml can't make
    # the workspace dir resolve outside the run root.
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str
    category: str
    difficulty: str = "hard"
    prompt: str
    visible_test_command: list[str] = ["python", "-m", "pytest", "-q"]
    timeout_s: int | None = None
    grader_timeout_s: int | None = None


class TaskDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suite: str
    spec: TaskSpec
    path: Path
    workspace_path: Path
    hidden_path: Path
    reference_path: Path

    @property
    def id(self) -> str:
        return self.spec.id


class AgentSpec(BaseModel):
    # YAML agent files come from users; a typo (`commnd`, `env_overide`) would
    # otherwise validate cleanly and run with default fields. Forbid extras so
    # mistakes surface at load time, matching TaskSpec.
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    provider: str
    model_id: str
    command: list[str]
    prompt_via_stdin: bool = False
    env_overrides: dict[str, str] = {}
    required_env: list[str] = []


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0


class UsageCapture(BaseModel):
    usage: TokenUsage = Field(default_factory=TokenUsage)
    reported_cost_usd: float | None = None
    reported_cost_source: str | None = None


class AgentExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int | None
    timed_out: bool
    latency_s: float
    stdout_path: str
    stderr_path: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    reported_cost_usd: float | None = None
    reported_cost_source: str | None = None


class TaskRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    agent_id: str
    agent_label: str
    model_id: str
    suite: str
    task_id: str
    task_title: str
    task_category: str
    success: bool | None
    partial_credit: float | None
    outcome_status: OutcomeStatus
    score: float
    checks: list[CheckResult]
    latency_s: float | None
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_in_by_model: dict[str, int] = Field(default_factory=dict)
    tokens_out_by_model: dict[str, int] = Field(default_factory=dict)
    reconstructed_per_task_cost_usd: float | None = None
    # Cost the agent CLI reported for *this single invocation*. Each (agent,
    # task) pair is a fresh subprocess, so what providers call a "session total"
    # is per-task here — the historical name `reported_run_total_cost_usd`
    # misled readers into thinking it was cumulative across the benchmark run.
    reported_session_cost_usd: float | None = None
    cost_provenance: CostProvenance = "cost_not_available"
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    agent_exit_code: int | None = None
    timed_out: bool = False
    workspace_path: str
    diff_path: str
    grader_path: str
    stdout_path: str
    stderr_path: str
    error: str | None = None
    # Set when the post-run integrity check found the agent *read* this task's
    # hidden grader or reference path — the run is contaminated and invalidated.
    cheating_detected: bool = False
    # Set when the agent *tried* to read the answer key but the sandbox blocked it.
    # The score stays trustworthy (the read failed); the attempt is recorded.
    attempted_grader_read: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


RUN_MANIFEST_SCHEMA_VERSION = 2

RunKind = Literal["live", "replay"]


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bump when fields are added/removed/renamed; `load_manifest` migrates
    # older versions in place and refuses forward-incompatible versions.
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str
    suite: str
    agent_ids: list[str]
    task_ids: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_s: float | None = None

    @field_validator("created_at", "completed_at")
    @classmethod
    def _ensure_aware_utc(cls, value: datetime | None) -> datetime | None:
        # A hand-written or older manifest may carry a naive datetime; comparing
        # naive vs aware raises TypeError and would crash `runs list`/compare on
        # one bad manifest. Treat naive timestamps as UTC.
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    repo_root: str
    run_dir: str
    default_agents: list[str]
    # "completed" if the run reached the end normally, "aborted" if --fail-fast
    # or a worker exception cut it short. Older manifests written before this
    # field existed load with the default and are indistinguishable from runs
    # that crashed mid-flight; that's the desired migration behaviour.
    status: RunStatus = "in_progress"
    # v2 fields (cross-run analysis). harness_git_sha captures heist's HEAD at
    # run start so `compare` can distinguish agent regression from harness
    # drift; None when capture failed (not a git checkout, git missing).
    harness_git_sha: str | None = None
    # Free-form labels assigned via the baseline registry or future tagging.
    tags: list[str] = Field(default_factory=list)
    # Provenance for run-replay (sibling proposal add-run-replay). Defensive
    # defaults live here so manifests stay valid regardless of which sibling
    # change lands first.
    source_run_id: str | None = None
    kind: RunKind = "live"
