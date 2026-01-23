.PHONY: help venv lock sync sync-frozen examples notebook test build clean distclean \
        bump bump-patch bump-minor bump-major \
        publish \
        release-patch release-minor release-major \
        show-version show-version-full \
        git-clean require-token

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
# Help
# --------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make venv              Create .venv"
	@echo "  make lock              Resolve deps -> uv.lock"
	@echo "  make sync              Sync env from uv.lock"
	@echo "  make sync-frozen       Sync strictly from uv.lock (CI-style)"
	@echo "  make examples          Install optional deps: stopro[examples]"
	@echo "  make notebook          Start Jupyter Lab in examples/"
	@echo "  make test              Run pytest"
	@echo "  make bump-patch        Bump version (patch)"
	@echo "  make bump-minor        Bump version (minor)"
	@echo "  make bump-major        Bump version (major)"
	@echo "  make build             Build wheel+sdist into dist/"
	@echo "  make publish           Publish to PyPI (requires token)"
	@echo "  make release-patch     Test, bump patch, commit, tag"
	@echo "  make release-minor     Test, bump minor, commit, tag"
	@echo "  make release-major     Test, bump major, commit, tag"
	@echo "  make clean             Remove build artifacts"
	@echo "  make distclean         Remove build artifacts + .venv"

# --------------------------------------------------------------------
# Environment / deps
# --------------------------------------------------------------------
# Optional: override the Python interpreter used to create the venv.
# Useful e.g. on macOS if you want a conda Python (OpenBLAS) instead of the
# default PyPI wheels (often Apple Accelerate).
#
# Usage:
#   make venv
#   make venv UV_PYTHON=$$(which python)
UV_PYTHON ?=

venv: .venv

.venv:
ifneq ($(strip $(UV_PYTHON)),)
	uv venv --python "$(UV_PYTHON)"
else
	uv venv
endif

lock:
	uv lock

sync: .venv
	uv sync

sync-frozen: .venv
	uv sync --frozen

# --------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------
git-clean:
	@git diff --quiet && git diff --cached --quiet || \
	 (echo "ERROR: git working tree is not clean"; exit 1)

require-token:
	@test -n "$$UV_PUBLISH_TOKEN" || \
	 (echo "ERROR: UV_PUBLISH_TOKEN missing (check .env)"; exit 1)

# --------------------------------------------------------------------
# Examples / notebooks
# --------------------------------------------------------------------
examples: sync
	uv pip install -e ".[examples]"

notebook: examples
	uv run jupyter lab examples

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
test: sync
	uv run pytest -q

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
# test -> bump -> commit -> tag
# Publishing is an explicit second step: `make publish`
# --------------------------------------------------------------------
release-patch: git-clean test bump-patch
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"

release-minor: git-clean test bump-minor
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"

release-major: git-clean test bump-major
	@ver=$$($(UV_VERSION_NUM_CMD)) && full=$$($(UV_VERSION_CMD)) && \
	git commit -am "Release $$full" && \
	git rev-parse "v$$ver" >/dev/null 2>&1 && \
	 (echo "ERROR: tag v$$ver already exists"; exit 1) || true && \
	git tag "v$$ver"