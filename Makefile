# Magpie developer entry points. `make help` lists them.
PY      ?= .venv/bin/python
UV      ?= uv
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: help install lint format test test-all smoke clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Create .venv (Python 3.12) with CPU torch and the dev/test extras
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) torch --index-url $(TORCH_CPU_INDEX)
	$(UV) pip install --python $(PY) -e ".[dev,hf,analysis,runners,data]"

lint:  ## Static checks (ruff)
	$(PY) -m ruff check .

format:  ## Auto-fix lint issues where safe
	$(PY) -m ruff check . --fix

test:  ## CPU test-suite (no GPU, no network)
	$(PY) -m pytest -m "not gpu and not net"

test-all:  ## Everything, including network-marked tests
	$(PY) -m pytest

smoke:  ## Push the T4 smoke-test kernel to Kaggle, wait, and pull results/
	$(PY) -m magpie.runners.kaggle smoke

clean:  ## Remove caches and build outputs
	rm -rf .pytest_cache .ruff_cache .kaggle-build build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
