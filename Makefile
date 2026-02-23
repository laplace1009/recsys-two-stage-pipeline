.PHONY: install lint test

install:
	uv sync --all-groups

lint:
	uv run ruff check src tests

test:
	uv run pytest
