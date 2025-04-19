setup:
	pip install uv==0.6.3
	uv venv
	uv sync

setup-test: setup
	uv sync --group test

test: setup-test
	uv run poc/cartpole-ppo.py

static: setup-test
	pyright .

check: setup-test
	ruff check src/ tests/
