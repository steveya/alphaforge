# Alphaforge

[![CI](https://github.com/steveya/alphaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/steveya/alphaforge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/alphaforge.svg)](https://pypi.org/project/alphaforge/)
[![Python](https://img.shields.io/pypi/pyversions/alphaforge.svg)](https://pypi.org/project/alphaforge/)
[![Docs](https://readthedocs.org/projects/alphaforge/badge/?version=latest)](https://alphaforge.readthedocs.io/en/latest/)

Composable point-in-time feature engineering and dataset building for systematic research.

Full documentation: <https://alphaforge.readthedocs.io/en/latest/>

## Installation

Install from PyPI:

```bash
pip install alphaforge
```

For development:

```bash
git clone https://github.com/steveya/alphaforge
cd alphaforge
pip install -e ".[dev]"
pre-commit install
```

## Quickstart

Run the local end-to-end MVP demo:

```bash
python examples/run_mvp_demo.py
```

## PIT Quick Example

```python
import pandas as pd
from alphaforge.pit import PITAccessor, PITTransformSpec
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

store = DuckDBParquetStore(root="./store")
pit = PITAccessor(store.conn())

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP", "GDP"],
            "obs_date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
            ],
            "value": [1.0, 1.1],
        }
    )
)

result = pit.apply_transform(
    PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff",
        axis="obs_path",
        op="diff",
        params={"periods": 1},
    ),
    overwrite=True,
)
print(result.transform_id, result.rows_written)
```

## Core Concepts

- `DataContext`: runtime wiring for sources, calendars, and store
- `FeatureTemplate` and `TargetTemplate`: fit/transform abstractions
- `DatasetSpec`: declarative dataset specification
- `build_dataset`: materialize, align, join, and return a dataset artifact
- `PITAccessor`: snapshot and revision-timeline access to revised series

## Development

- Tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy alphaforge`
- Docs: `mkdocs build --strict`
- Build: `python -m build`

## License

MIT
