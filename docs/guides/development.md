# Development Guide

## Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quality checks

```bash
ruff check .
mypy alphaforge
pytest
```

## Core platform regression gates

When a change touches canonical PIT, data-context, dataset, or operational
surfaces from the roadmap, also run:

```bash
python -m pytest tests/contracts
python -m benchmarks.pit
```

## Public web contributors

For the public loader pack under `alphaforge.data.public_web`, use the
dedicated [public-web source authoring guide](public-web-source-authoring.md).
It covers helper-family selection, registry/export wiring, targeted test
expectations, and the required docs plus Linear plan updates.

## Build package

```bash
python -m build
python -m twine check dist/*
```

## Release flow

1. Update changelog.
2. Create a tag `vX.Y.Z`.
3. Push the tag.
4. GitHub Actions release workflow publishes to PyPI.
