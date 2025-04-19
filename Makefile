setup:
	pip install uv
	uv venv
	uv pip install -e .

test: setup
	uv sync --group test
	uv run poc/cartpole-ppo.py

static:
	npm install -i basedpyright
	pyright .

check: setup
	ruff check src/ tests/
