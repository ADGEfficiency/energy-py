setup:
	pip install uv
	uv venv
	uv sync

test: setup
	uv sync --group test
	uv run poc/cartpole-ppo.py

static:
	uv sync --group test
	pyright .

check: setup
	ruff check src/ tests/
