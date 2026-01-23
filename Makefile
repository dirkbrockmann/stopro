.PHONY: help venv lock sync sync-frozen examples notebook test build publish-test publish clean distclean \
        bump bump-patch bump-minor bump-major release-patch release-minor release-major

help:
	@echo "Targets:"
	@echo "  make venv           Create .venv"
	@echo "  make lock           Resolve deps -> uv.lock"
	@echo "  make sync           Sync env from uv.lock"
	@echo "  make sync-frozen    Sync strictly from uv.lock (no resolving; good for CI)"
	@echo "  make examples       Install optional deps: stopro[examples]"
	@echo "  make notebook       Start Jupyter Lab in examples/"
	@echo "  make test           Run pytest"
	@echo "  make bump-patch     Bump version (patch) in pyproject.toml"
	@echo "  make bump-minor     Bump version (minor) in pyproject.toml"
	@echo "  make bump-major     Bump version (major) in pyproject.toml"
	@echo "  make build          Build wheel+sdist into dist/"
	@echo "  make publish-test   Publish to TestPyPI"
	@echo "  make publish        Publish to PyPI"
	@echo "  make release-patch  Test, bump patch, tag, build, publish"
	@echo "  make release-minor  Test, bump minor, tag, build, publish"
	@echo "  make release-major  Test, bump major, tag, build, publish"
	@echo "  make clean          Remove build artifacts"
	@echo "  make distclean      Remove build artifacts + .venv"

venv: .venv
.venv:
	uv venv

lock:
	uv lock

sync: .venv
	uv sync

sync-frozen: .venv
	uv sync --frozen

# Install extras for running notebooks/examples (published extras, not a local dependency-group)
examples: sync
	uv pip install -e ".[examples]"

notebook: examples
	uv run jupyter lab examples

test: sync
	uv run pytest -q

clean:
	rm -rf dist build

distclean: clean
	rm -rf .venv

# Version bumping (npm-like)
bump: bump-patch

bump-patch:
	uv version --bump patch

bump-minor:
	uv version --bump minor

bump-major:
	uv version --bump major

build: clean sync
	uv build

publish-test: build
	uv publish --repository testpypi

publish: build
	uv publish

release-patch: test bump-patch
	@ver=$$(uv version) && \
	git commit -am "Release $$ver" && \
	git tag "v$$ver" && \
	uv build && \
	uv publish

release-minor: test bump-minor
	@ver=$$(uv version) && \
	git commit -am "Release $$ver" && \
	git tag "v$$ver" && \
	uv build && \
	uv publish

release-major: test bump-major
	@ver=$$(uv version) && \
	git commit -am "Release $$ver" && \
	git tag "v$$ver" && \
	uv build && \
	uv publish