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
