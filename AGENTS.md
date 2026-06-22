# AGENTS.md — HEIST

HEIST is a local benchmark harness for CLI coding agents. It runs agents against
task packs in isolated workspaces and grades the results with strict JSON graders.

User-level guidance (tone, principles, git etiquette, Python defaults) lives in
`~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated
here. This file is for project-specific facts.

## Layout

```
src/heist/                      Harness code (CLI, runner, graders, models)
tasks/<suite>/<task-id>/        Benchmark fixtures — four-file contract:
├── task.yaml                   Validated task metadata + prompt
├── workspace/                  Copied into the agent-visible run workspace
├── hidden/grader.py            Strict JSON grader, hidden from the agent
└── reference/                  Known-good solution; tests use it to prove solvability
runs/                           Generated run output (not source)
```

See [src/heist/cli.py](src/heist/cli.py) for the entrypoint and
[tasks/examples/](tasks/examples/) for worked task packs.

## Rules

- Treat task packs as benchmark fixtures. Do not edit `hidden/` or `reference/`
  unless the change is explicitly about task authoring, calibration, or grading.
- Preserve public return shapes named in task prompts. Hidden graders should
  test behavior, not incidental implementation details.
- `runs/` is generated output. Keep run artifacts out of source changes.
- Do not add network calls, local secrets, or machine-specific assumptions to
  graders or task workspaces.

## Checks

For harness changes (matches CI in [.github/workflows/ci.yml](.github/workflows/ci.yml)):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -q
```

For task-contract changes, also:

```bash
uv run pytest tests/test_tasks.py
uv run heist tasks list --suite examples
```

Before calling work complete, inspect the files you changed and check the diff
for accidental fixture churn. If a doc here disagrees with the code, fix the doc
in the same change.
