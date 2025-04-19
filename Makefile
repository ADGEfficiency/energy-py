setup:
	uv pip install -e .
	pip install uv

test: setup
	uv sync --group test
	uv run poc/cartpole-ppo.py

static:
	npm install -i basedpyright
	pyright .

check: setup
	ruff check src/ tests/
