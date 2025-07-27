# Project: research_tool_rag

PYTHON=python3
ENV_DIR=venv

export PYTHONPATH=src

# --- Environment Management ---

$(ENV_DIR):
	$(PYTHON) -m venv $(ENV_DIR)

print-install-message:
	@echo "✅ Virtual environment created at $(ENV_DIR)"
	@echo "💡 Run the following to activate:"
	@echo "   source $(ENV_DIR)/bin/activate"

env: $(ENV_DIR) print-install-message

# --- Formatting and Linting ---


fixup:
	@if [ -d src ]; then \
	autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports -r src; \
	isort src/; \
	black --line-length=100 src/; \
  fi
	@if [ -d tests ]; then \
	autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports -r tests; \
	isort tests/; \
	black --line-length=100 tests/; \
  fi
	@if [ -d sandbox ]; then \
	isort sandbox/; \
	black --line-length=100 sandbox/; \
  fi

lint:
	isort --check-only src tests sandbox
	black --check --line-length=100 src tests sandbox
	flake8 src tests sandbox
	mypy src

# --- Testing ---

test:
	pytest -vv tests/

# --- Code & Data Pipelines ---

ingest:
	python src/research_tool_rag/rag/ingest_pipeline.py


run-agent:
	python src/research_tool_rag/rag/main.py

# --- Combined Actions ---

commit: fixup lint test

# --- Maintenance ---

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info

help:
	@echo ""
	@echo "🛠️  research_tool_rag Makefile Commands:"
	@echo "  make env         - Create virtual environment"
	@echo "  make fixup       - Autofix imports and formatting"
	@echo "  make lint        - Run linters (isort, black, ruff)"
	@echo "  make test        - Run pytest"
	@echo "  make ingest      - Run RAG chunking + store in Qdrant"
	@echo "  make run-agent   - Run the full agent pipeline"
	@echo "  make commit      - Run fixup, lint, and tests (pre-commit style)"
	@echo "  make clean       - Remove pycache and build files"
	@echo ""
