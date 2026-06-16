.PHONY: lint format fix check leaderboard report figures board

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

fix:
	uv run ruff check --fix . && uv run ruff format .

check: lint format
	uv run pytest tests/ -q

# Sanitize a run's results.jsonl into the committed, leak-free leaderboard.
# RUN overrides the source run dir (default: runs/frontier-2026-06-15).
RUN ?= runs/frontier-2026-06-15
leaderboard:
	uv run python scripts/make_leaderboard.py --run $(RUN)

# Re-render the committed summary.md + report.html from a run's results.jsonl.
report:
	uv run python scripts/make_report.py --run $(RUN)

# Redraw figures/*.jpg from the committed leaderboard. Needs the figures group.
figures:
	uv run --group figures python scripts/make_figures.py

# Regenerate every committed board artifact from $(RUN), in dependency order.
board: leaderboard report figures
