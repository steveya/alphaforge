# Quickstart: Build a Dataset

This example uses a dummy source with two feature families and one target.

```python
import numpy as np
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    FeatureRequest,
    JoinPolicy,
    MissingnessPolicy,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)
from alphaforge.features.target_template import TargetFrame
from alphaforge.store.local_parquet import LocalParquetStore
from alphaforge.time.calendar import TradingCalendar

from examples.dummy_source import DummySource
from examples.features_lag_returns import LagReturnsTemplate
from examples.features_macro_carry import MacroCarryTemplate

cal = TradingCalendar("XNYS", tz="UTC")
dates = cal.sessions("2020-01-01", "2020-03-31")
entities = ["AAA", "BBB"]

rng = np.random.default_rng(123)
rows = []
for entity in entities:
    px = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
    for date, price in zip(dates, px):
        rows.append({"date": date, "entity_id": entity, "close": float(price)})

ohlcv = pd.DataFrame(rows)
macro = pd.DataFrame(
    [
        {"date": pd.Timestamp("2020-01-31"), "entity_id": "CPI", "value": 1.0},
        {"date": pd.Timestamp("2020-02-29"), "entity_id": "CPI", "value": 2.0},
        {"date": pd.Timestamp("2020-03-31"), "entity_id": "CPI", "value": 3.0},
    ]
)

ctx = DataContext(
    sources={"dummy": DummySource(ohlcv_long=ohlcv, macro_long=macro)},
    calendars={"XNYS": cal},
    store=LocalParquetStore("./alphaforge_demo_store"),
)

features = [
    FeatureRequest(
        template=LagReturnsTemplate(),
        params={"lags": 5, "source": "dummy", "table": "market.ohlcv", "price_col": "close"},
    ),
    FeatureRequest(
        template=MacroCarryTemplate(),
        params={"source": "dummy", "table": "macro.series", "value_col": "value", "method": "ffill"},
    ),
]


class NextDaySqLogRetTarget:
    name = "target_nextday_sqret"
    version = "1.0"
    param_space = {}

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice, state):
        panel = ctx.fetch_panel(
            "dummy",
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


spec = DatasetSpec(
    universe=UniverseSpec(entities=entities),
    time=TimeSpec(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-03-31"), calendar="XNYS", grid="B"),
    features=features,
    target=TargetRequest(template=NextDaySqLogRetTarget(), params={}, horizon=1, name="y"),
    join_policy=JoinPolicy(how="inner", sort_index=True),
    missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    name="demo_dataset",
)

artifact = build_dataset(ctx, spec, persist=True)
print(artifact.X.shape)
print(int(artifact.y.notna().sum()))
```

See [Dataset Spec guide](../guides/dataset-spec.md) for full options.
