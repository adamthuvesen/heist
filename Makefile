.PHONY: lint format fix check public-report figures board

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

fix:
	uv run ruff check --fix . && uv run ruff format .

check: lint format
	uv run python -m pytest tests/ -q

# Private source checkout and run dir. The held-out difficulty tier map stays in
# the private repo; only the aggregate public report and PNGs are committed here.
PRIVATE_REPO ?= ../heist
RUN ?= $(PRIVATE_REPO)/runs/frontier-2026-06-15
REPORT ?= results/frontier-2026-06-15/report.html

public-report:
	uv run --project $(PRIVATE_REPO) python $(PRIVATE_REPO)/scripts/make_public_report.py --run $(RUN) --out $(abspath $(REPORT))

# Redraw the README PNGs from the sanitized report payload. Needs figures group.
figures:
	uv run --group figures python scripts/make_figures.py --report $(REPORT)

# Regenerate every committed public artifact from $(RUN), in dependency order.
board: public-report figures
