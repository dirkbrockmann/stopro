.PHONY: help venv lock sync sync-frozen sync-dev examples notebook test build clean distclean \
        bump bump-patch bump-minor bump-major \
        publish \
        release-patch release-minor release-major \
        show-version show-version-full \
        git-clean require-token require-remote push-release \
        bench bench-list bench-%

# --------------------------------------------------------------------
# Load local environment variables if present (do NOT commit .env)
# --------------------------------------------------------------------
ifneq (,$(wildcard .env))
  include .env
  export
endif

# --------------------------------------------------------------------
# Version helpers
# uv prints: "<name> <version>" (e.g. "stopro 0.3.5")
# Do NOT store versions in Make variables (they would go stale after bump)
# --------------------------------------------------------------------
UV_VERSION_CMD = uv version
UV_VERSION_NUM_CMD = uv version | awk '{print $$NF}'

# Expected env vars (loaded from .env):
# - UV_PUBLISH_TOKEN        (PyPI)

# --------------------------------------------------------------------
# Python interpreter for venv creation
# Default is portable; override as needed, e.g.:
#   make PYTHON=/opt/homebrew/bin/python3 venv
# --------------------------------------------------------------------
PYTHON ?= python3

# --------------------------------------------------------------------
# Help
# --------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make venv              Create .venv"
	@echo "  make lock              Resolve deps -> uv.lock (maintainers)"
	@echo "  make sync              Sync env strictly from uv.lock (recommended default)"
	@echo "  make sync-dev          Sync env allowing lock refresh (maintainers)"
	@echo "  make sync-frozen       Alias for sync"
	@echo "  make examples          Sync + install optional deps: stopro[examples]"
	@echo "  make notebook          Start Jupyter Lab in examples/"
	@echo "  make test              Run pytest"
	@echo "  make bench             Run all benchmarks"
	@echo "  make bench-list        List available benchmarks"
	@echo "  make bench-wiener      Run a single benchmark (example)"
	@echo "  make bump-patch        Bump version (patch)"
	@echo "  make bump-minor        Bump version (minor)"
	@echo "  make bump-major        Bump version (major)"
	@echo "  make build             Build wheel+sdist into dist/"
	@echo "  make publish           Publish to PyPI (requires token)"
	@echo "  make release-patch     Test, bump patch, commit, tag, push"
	@echo "  make release-minor     Test, bump minor, commit, tag, push"
	@echo "  make release-major     Test, bump major, commit, tag, push"
	@echo "  make push-release      Push current branch + tags"
	@echo "  make clean             Remove build artifacts"
	@echo "  make distclean         Remove build artifacts + .venv"

# --------------------------------------------------------------------
# Environment / deps
# --------------------------------------------------------------------
venv: .venv

.venv:
	uv venv --python $(PYTHON)

# Maintainers only: refresh resolution into uv.lock
lock:
	uv lock

# Default path for collaborators / HPC: strictly use uv.lock as-is
sync: .venv
	uv sync --frozen

# Alias for clarity / backwards compatibility
sync-frozen: sync

# Maintainers only: allow uv to re-lock before syncing if needed
sync-dev: .venv
	uv sync

# --------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------
git-clean:
	@git diff --quiet && git diff --cached --quiet || \
	 (echo "ERROR: git working tree is not clean"; exit 1)

require-token:
	@test -n "$$UV_PUBLISH_TOKEN" || \
	 (echo "ERROR: UV_PUBLISH_TOKEN missing (check .env)"; exit 1)

require-remote:
	@git remote get-url origin >/dev/null 2>&1 || \
	 (echo "ERROR: remote 'origin' not configured"; exit 1)
	@git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1 || \
	 (echo "ERROR: no upstream branch set. Run: git push -u origin $$(git branch --show-current)"; exit 1)

push-release: require-remote
	@git push
	@git push --tags

# --------------------------------------------------------------------
# Examples / notebooks
# --------------------------------------------------------------------
examples: .venv
	uv sync --frozen --extra examples

notebook: examples
	@mkdir -p .jupyter .jupyter_data .jupyter_runtime
	JUPYTER_CONFIG_DIR="$(PWD)/.jupyter" \
	JUPYTER_DATA_DIR="$(PWD)/.jupyter_data" \
	JUPYTER_RUNTIME_DIR="$(PWD)/.jupyter_runtime" \
	uv run jupyter lab examples

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
test: sync
	uv run pytest -q

# --------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------
bench: sync
	uv run python -m benchmarks.run all

bench-list: sync
	uv run python -m benchmarks.run --list

bench-%: sync
	uv run python -m benchmarks.run $*

# --------------------------------------------------------------------
# Version bumping (npm-style)
# --------------------------------------------------------------------
bump: bump-patch

bump-patch:
	uv version --bump patch

bump-minor:
	uv version --bump minor

bump-major:
	uv version --bump major

show-version:
	@$(UV_VERSION_NUM_CMD)

show-version-full:
	@$(UV_VERSION_CMD)

# --------------------------------------------------------------------
# Build / clean
# --------------------------------------------------------------------
clean:
	rm -rf dist build

distclean: clean
	rm -rf .venv

build: clean sync
	uv build

# --------------------------------------------------------------------
# Publish (LOCAL ONLY, token via .env)
# --------------------------------------------------------------------
publish: build require-token
	uv publish --token "$$UV_PUBLISH_TOKEN"

# --------------------------------------------------------------------
# Release targets
# test -> bump -> commit -> tag -> push
# Publishing remains explicit: `make publish`
# --------------------------------------------------------------------
release-patch: git-clean test bump-patch
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"
	@$(MAKE) push-release

release-minor: git-clean test bump-minor
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"
	@$(MAKE) push-release

release-major: git-clean test bump-major
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"
	@$(MAKE) push-release