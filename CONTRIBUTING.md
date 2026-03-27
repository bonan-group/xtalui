# Contributing

## Local Setup

```bash
uv venv --python /usr/bin/python3 .venv
uv sync --extra dev
```

## Checks

Run the same checks as CI before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

## Formatting

To apply Ruff formatting locally:

```bash
uv run ruff format .
```

## Release Flow

1. Update `pyproject.toml` version.
2. Update `CHANGELOG.md`.
3. Commit the release changes.
4. Create and push a SemVer tag like `v0.1.1`.

```bash
git tag v0.1.1
git push origin v0.1.1
```

The GitHub release workflow will verify the tag/version match, build the package, create the GitHub Release, and publish to PyPI.
