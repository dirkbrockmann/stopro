.PHONY: help venv lock sync sync-frozen examples notebook test build clean distclean \
        bump bump-patch bump-minor bump-major \
        publish-test publish \
        release-patch release-minor release-major \
        release-patch-test release-minor-test release-major-test \
        show-version show-version-full

# uv prints: "<name> <version>" (e.g. "stopro 0.3.5").
# This extracts just the numeric version for tags etc.
UV_VERSION = $(shell uv version | awk '{print $$NF}')
UV_VERSION_FULL = $(shell uv version)

help:
	@echo "Targets:"
	@echo "  make venv              Create .venv"
	@echo "  make lock              Resolve deps -> uv.lock"
	@echo "  make sync              Sync env from uv.lock"
	@echo "  make sync-frozen       Sync strictly from uv.lock (no resolving; good for CI)"
	@echo "  make examples          Install optional deps: stopro[examples]"
	@echo "  make notebook          Start Jupyter Lab in examples/"
	@echo "  make test              Run pytest"
	@echo "  make bump-patch        Bump version (patch) in pyproject.toml"
	@echo "  make bump-minor        Bump version (minor) in pyproject.toml"
	@echo "  make bump-major        Bump version (major) in pyproject.toml"
	@echo "  make build             Build wheel+sdist into dist/"
	@echo "  make publish-test      Publish to TestPyPI"
	@echo "  make publish           Publish to PyPI"
	@echo "  make release-patch     Test, bump patch, commit, tag, build, publish"
	@echo "  make release-minor     Test, bump minor, commit, tag, build, publish"
	@echo "  make release-major     Test, bump major, commit, tag, build, publish"
	@echo "  make clean             Remove build artifacts"
	@echo "  make distclean         Remove build artifacts + .venv"

# Create the venv directory (idempotent)
venv: .venv
.venv:
	uv venv

lock:
	uv lock

sync: .venv
	uv sync

sync-frozen: .venv
	uv sync --frozen

# Install optional deps for notebooks/examples (published extras, not a dependency-group)
examples: sync
	uv pip install -e ".[examples]"

notebook: examples
	uv run jupyter lab examples

test: sync
	uv run pytest -q

# Version bumping (npm-like)
bump: bump-patch

bump-patch:
	uv version --bump patch

bump-minor:
	uv version --bump minor

bump-major:
	uv version --bump major

show-version:
	@echo $(UV_VERSION)

show-version-full:
	@echo $(UV_VERSION_FULL)

clean:
	rm -rf dist build

distclean: clean
	rm -rf .venv

build: clean sync
	uv build

publish-test: build
	uv publish --repository testpypi

publish: build
	uv publish

# Release targets: test -> bump -> commit -> tag -> build -> publish
# Commit message includes the full uv version output (e.g. "stopro 0.3.5") for readability.
# Tag uses numeric version only (e.g. "v0.3.5") to avoid spaces/invalid tag names.
release-patch: test bump-patch
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish

release-minor: test bump-minor
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish

release-major: test bump-major
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish

# Same, but publish to TestPyPI
release-patch-test: test bump-patch
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish --repository testpypi

release-minor-test: test bump-minor
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish --repository testpypi

release-major-test: test bump-major
	git commit -am "Release $(UV_VERSION_FULL)"
	git tag "v$(UV_VERSION)"
	uv build
	uv publish --repository testpypi