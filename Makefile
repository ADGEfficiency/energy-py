setup:
	curl -LsSf https://astral.sh/uv/0.6.3/install.sh | sh
	uv venv
	uv sync

setup-test: setup
	uv sync --group test

test: setup-test
	uv run poc/cartpole-ppo.py
	uv run poc/battery.py

STATIC_LEVEL=warning # error or warning
static: setup-test
	uv run basedpyright src --level $(STATIC_LEVEL)

check: setup-test
	uv run ruff check src
