setup:
	curl -LsSf https://astral.sh/uv/0.6.3/install.sh | sh
	uv venv
	uv sync

setup-test: setup
	uv sync --group test

test: setup-test
	uv run examples/battery.py

static: setup-test
	uv run basedpyright src poc

check: setup-test
	uv run ruff check src poc
