# Alphaforge

Composable feature engineering and dataset building for systematic research.

Alphaforge provides a small set of primitives for:
- Defining feature and target templates (fit/transform style, parameterized)
- Fetching data from arbitrary sources into a common panel shape
- Aligning/merging features on a calendar-aware grid
- Building reproducible, scalable datasets from a declarative specification

This README focuses on the new scalable Dataset API: `DatasetSpec` and `build_dataset`.

## Installation

Clone and install in editable mode:

```
git clone https://github.com/steveya/alphaforge
cd alphaforge
pip install -e .
```

Optional: for the examples below you only need pandas/numpy which are standard dependencies.

## Core concepts

- DataContext: runtime wiring of data sources, calendars, and materialization store
- FeatureTemplate: a template with a `transform(ctx, params, slice, state)` that returns a FeatureFrame
- TargetTemplate: same pattern but returns a TargetFrame (y series + meta)
- SliceSpec: time window, entities, grid, and asof (point-in-time) cutoff
- DatasetSpec: declarative spec of universe, time, features (arbitrary list), target, and build policies
- build_dataset: materializes features/target, joins them, applies missingness policy, and returns a DatasetArtifact

## Quickstart: build a dataset

Below is a minimal, end-to-end example using an in-memory DummySource and two example feature templates. It demonstrates:
- Multiple feature families
- A simple target builder (next-day squared log return)
- Join and missingness policies

```python
import numpy as np
import pandas as pd

from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext
from alphaforge.store.local_parquet import LocalParquetStore

from alphaforge.features.dataset_spec import (
	DatasetSpec,
	UniverseSpec,
	TimeSpec,
	FeatureRequest,
	TargetRequest,
	SliceOverride,
	JoinPolicy,
	MissingnessPolicy,
)
from alphaforge.features.target_template import TargetFrame, TargetTemplate
from alphaforge.features.dataset_builder import build_dataset

# Example pieces from the repo's examples/
from examples.dummy_source import DummySource
from examples.features_lag_returns import LagReturnsTemplate
from examples.features_macro_carry import MacroCarryTemplate


# 1) Build a DataContext (sources + calendars + store)
cal = TradingCalendar("XNYS", tz="UTC")
dates = cal.sessions("2020-01-01", "2020-03-31")
entities = ["AAA", "BBB"]

# synthetic daily closes
rng = np.random.default_rng(123)
rows = []
for e in entities:
	px = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
	for d, p in zip(dates, px):
		rows.append({"date": d, "entity_id": e, "close": float(p)})
ohlcv = pd.DataFrame(rows)

# synthetic monthly macro
macro = pd.DataFrame(
	[
		{"date": pd.Timestamp("2020-01-31"), "entity_id": "CPI", "value": 1.0},
		{"date": pd.Timestamp("2020-02-29"), "entity_id": "CPI", "value": 2.0},
		{"date": pd.Timestamp("2020-03-31"), "entity_id": "CPI", "value": 3.0},
	]
)

src = DummySource(ohlcv_long=ohlcv, macro_long=macro)
store = LocalParquetStore("./alphaforge_demo_store")
ctx = DataContext(sources={"dummy": src}, calendars={"XNYS": cal}, store=store)


# 2) Define features using FeatureRequest
features = (
	FeatureRequest(
		template=LagReturnsTemplate(),
		params={"lags": 5, "source": "dummy", "table": "market.ohlcv", "price_col": "close"},
	),
	FeatureRequest(
		template=MacroCarryTemplate(),
		params={"source": "dummy", "table": "macro.series", "value_col": "value", "method": "ffill"},
	),
)


# 3) Define a simple TargetTemplate (next-day squared log return)
class NextDaySqLogRetTarget:
	name = "target_nextday_sqret"
	version = "1.0"
	param_space = {}

	def fit(self, ctx, params, fit_slice):
		return None

	def transform(self, ctx, params, slice, state):
		panel = ctx.fetch_panel(
			"dummy",
			# fetch just close from market.ohlcv
			Query(
				table="market.ohlcv",
				columns=["close"],
				start=slice.start,
				end=slice.end,
				entities=slice.entities,
				asof=slice.asof,
				grid=slice.grid,
			),
		)
		px = panel.df["close"].astype(float)
		logret = np.log(px).groupby(level="entity_id").diff()
		y = (logret.groupby(level="entity_id").shift(-1) ** 2).rename("y")
		return TargetFrame(y=y, meta={"definition": "(logret_{t+1})^2"})


target = TargetRequest(template=NextDaySqLogRetTarget(), params={}, horizon=1, name="y")


# 4) Build DatasetSpec and materialize
spec = DatasetSpec(
	universe=UniverseSpec(entities=entities),
	time=TimeSpec(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-03-31"), calendar="XNYS", grid="B"),
	features=list(features),
	target=target,
	join_policy=JoinPolicy(how="inner", sort_index=True),
	missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
	name="demo_dataset",
	tags={"example": True},
)

artifact = build_dataset(ctx, spec, persist=True)
print("X shape:", artifact.X.shape)
print("y non-missing:", int(artifact.y.notna().sum()))
print("catalog rows:", len(artifact.catalog))
```

That’s it. You can add arbitrarily many feature requests. Each request can override slice parameters (lookback, grid, asof) via `SliceOverride` when needed, for example to extend fetch windows for lagged features:

```python
FeatureRequest(
	template=LagReturnsTemplate(),
	params={"lags": 20, "source": "dummy", "table": "market.ohlcv", "price_col": "close"},
	slice_override=SliceOverride(lookback=pd.Timedelta(days=60)),
)
```

## API reference (key types)

- DatasetSpec
  - universe: UniverseSpec(entities)
  - time: TimeSpec(start, end, calendar="XNYS", grid="B", asof=None)
  - features: list[FeatureRequest]
  - target: TargetRequest
  - join_policy: JoinPolicy(how="inner"|"outer", sort_index=True)
  - missingness: MissingnessPolicy(final_row_policy="drop_if_any_nan"|"keep")
  - name, tags

- FeatureRequest
  - template: a FeatureTemplate instance with .transform returning FeatureFrame
  - params: dict of template parameters
  - slice_override: optional SliceOverride(lookback, grid, asof)

- TargetRequest
  - template: a TargetTemplate instance returning TargetFrame (y + meta)
  - params: dict of template parameters
  - horizon, name, slice_override

- build_dataset(ctx, spec, persist=True) -> DatasetArtifact
  - X: pd.DataFrame (MultiIndex: [ts_utc, entity_id])
  - y: pd.Series (aligned to X index; if single-index, expanded to entities)
  - catalog: pd.DataFrame (feature catalog)
  - meta/aux: dictionaries for metadata and optional extras

## Examples

The `examples/` directory contains:
- `run_mvp_demo.py`: feature materialization and joins (pre-dataset-builder)
- `features_lag_returns.py`, `features_macro_carry.py`, `dummy_source.py`: building blocks used above

## Integrations

The [volatility-forecast](https://github.com/steveya/volatility-forecast) project demonstrates a domain-specific package built on Alphaforge. It defines feature families and targets (e.g., lagged returns, next-day squared return) and uses the scalable `DatasetSpec` via a convenience `VolDatasetSpec` wrapper in its pipeline.

## License

MIT