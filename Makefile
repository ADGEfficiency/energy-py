setup:
	curl -LsSf https://astral.sh/uv/0.6.3/install.sh | sh
	uv venv
	uv sync

clean:
	rm -rf ./data/tensorboard/

TB_DIR=./data/tensorboard/
monitor:
	uv run tensorboard --logdir $(TB_DIR) --bind_all

setup-test: setup
	uv sync --group test

test: setup-test
	uv run pytest tests --tb=short -p no:warnings --disable-warnings --cov=src --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html:htmlcov

test-examples: setup-test
	uv run examples/dataset.py
	uv run examples/battery.py

SRC_DIRS=src examples
static: setup-test
	uv run basedpyright $(SRC_DIRS) --level error

RUFF_ARGS=
check: setup-test
	uv run ruff check $(SRC_DIRS) $(RUFF_ARGS)
