.PHONY: help venv lock sync sync-frozen examples test build clean distclean publish-test publish

help:
	@echo "Targets:"
	@echo "  make venv         Create .venv"
	@echo "  make lock         Resolve deps -> uv.lock"
	@echo "  make sync         Sync base env from uv.lock (includes dev group)"
	@echo "  make sync-frozen  Sync strictly from uv.lock (no resolving; good for CI)"
	@echo "  make examples     Sync env incl. examples group"
	@echo "  make test         Run pytest"
	@echo "  make build        Build wheel+sdist into dist/"
	@echo "  make publish-test Upload dist/* to TestPyPI"
	@echo "  make publish      Upload dist/* to PyPI"
	@echo "  make clean        Remove build artifacts"
	@echo "  make distclean    Remove build artifacts + .venv"

# Create the venv directory (idempotent)
venv: .venv

.venv:
	uv venv

lock:
	uv lock

# Base sync (dev group included by default)
sync: .venv
	uv sync

# Strict sync (fails if lockfile is out of date)
sync-frozen: .venv
	uv sync --frozen

# Install the examples group in addition to base/dev
examples: .venv
	uv sync --group examples

notebook: examples
	uv run jupyter lab examples

test: sync
	uv run pytest -q

build: clean sync
	uv run python -m build

publish-test: build
	uv run python -m twine upload --repository testpypi dist/*

publish: build
	uv run python -m twine upload dist/*

clean:
	rm -rf build dist
	rm -rf *.egg-info src/*.egg-info
	@# If you ever see nested egg-info, this catches it on mac/linux:
	find src -name "*.egg-info" -maxdepth 5 -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv