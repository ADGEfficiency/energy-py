setup:
	curl -LsSf https://astral.sh/uv/0.6.3/install.sh | sh
	uv venv
	uv sync

setup-test: setup
	uv sync --group test

test: setup-test
	uv run examples/dataset.py
	uv run examples/battery.py

SRC_DIRS=src examples
static: setup-test
	uv run basedpyright $(SRC_DIRS) --level error

check: setup-test
	uv run ruff check $(SRC_DIRS)
