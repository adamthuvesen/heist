# AGENTS.md

Repo-local instructions for coding agents working in HEIST.

## Project Shape

HEIST is a local benchmark harness for CLI coding agents. The harness code
lives in `src/heist/`; benchmark fixtures live in `tasks/<suite>/<task-id>/`.

Each task has the same contract:

- `task.yaml`: validated task metadata and prompt
- `workspace/`: copied into the agent-visible run workspace
- `hidden/grader.py`: strict JSON grader, not visible to the benchmarked agent
- `reference/`: known-good implementation used by tests to prove solvability

## Rules

- Treat task packs as benchmark fixtures. Do not edit `hidden/` or `reference/`
  unless the change is explicitly about task authoring, calibration, or grading.
- Preserve public return shapes named in task prompts. Hidden graders should
  test behavior, not incidental implementation details.
- Keep run artifacts out of source changes. `runs/` is generated output.
- Use `uv` for local commands.
- Prefer `pathlib.Path`, typed Python, Pydantic validation, and the style already
  present in neighboring files.
- Do not add network calls, local secrets, or machine-specific assumptions to
  graders or task workspaces.

## Checks

For harness changes:

```bash
uv run pytest
uv run ruff check .
```

For task-contract changes:

```bash
uv run pytest tests/test_tasks.py
uv run heist tasks list --suite examples
```

Before calling work complete, inspect the files you changed and check the diff
for accidental fixture churn.

