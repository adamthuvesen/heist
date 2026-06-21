.PHONY: lint format fix check figures

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

fix:
	uv run ruff check --fix . && uv run ruff format .

check: lint format
	uv run python -m pytest tests/ -q

# Redraw README figures from the committed leak-safe report payload.
REPORT ?= results/frontier-2026-06-15/report.html

figures:
	uv run --group figures python scripts/make_figures.py --report $(REPORT)
