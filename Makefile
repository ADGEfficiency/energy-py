test:
	uv sync --group test
	uv run poc/cartpole-ppo.py

static:
	pyright .
