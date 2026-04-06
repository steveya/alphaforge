# Quickstart: Build a Dataset

This example uses the canonical adapter path plus built-in notebook-ready
market templates.

```python
import numpy as np
import pandas as pd

from alphaforge.data.adapter import SourceAdapterBase
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.data.types import FetchResult
from alphaforge.features import LagReturnsTemplate, RollingVolatilityTemplate
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    FeatureRequest,
    FeatureRequestGroup,
    JoinPolicy,
    MissingnessPolicy,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)
from alphaforge.features.target_template import TargetFrame
from alphaforge.features.template import SliceSpec
from alphaforge.time.calendar import TradingCalendar


class InMemoryMarketAdapter(SourceAdapterBase):
    source_name = "market"
    datasets = frozenset({"market.ohlcv"})

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def fetch(self, query: Query, *, max_staleness=None) -> FetchResult:
        frame = self._frame[self._frame["series_key"].isin(query.entities)].copy()
        obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.start is not None:
            frame = frame[obs >= query.start]
            obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.end is not None:
            frame = frame[obs <= query.end]

        keep = ["series_key", "obs_date"] + list(query.columns)
        return FetchResult(
            data=frame[keep].reset_index(drop=True),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
        )

    def list_entities(self, dataset: str) -> list[str]:
        return sorted(self._frame["series_key"].unique())


class NextDaySqLogRetTarget:
    name = "target_nextday_sqret"
    version = "1.0"
    param_space = {}

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice: SliceSpec, state):
        result = ctx.load(
            "market.ohlcv",
            columns=["close"],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source="market",
        )
        frame = result.data.copy()
        calendar = ctx.calendars["XNYS"]
        frame["ts_utc"] = [
            calendar.session_close_utc(ts)
            for ts in pd.to_datetime(frame["obs_date"], utc=True)
        ]
        prices = (
            frame.set_index(["ts_utc", "series_key"])["close"]
            .rename_axis(index=["ts_utc", "entity_id"])
            .sort_index()
            .astype(float)
        )
        logret = np.log(prices).groupby(level="entity_id").diff()
        y = (logret.groupby(level="entity_id").shift(-1) ** 2).rename("y")
        return TargetFrame(y=y, meta={"definition": "(logret_{t+1})^2"})


cal = TradingCalendar("XNYS", tz="UTC")
dates = cal.sessions("2024-01-02", "2024-02-16")
entities = ["AAA", "BBB"]

rng = np.random.default_rng(123)
rows = []
for entity in entities:
    prices = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
    for obs_date, close in zip(dates, prices, strict=True):
        rows.append({"series_key": entity, "obs_date": obs_date, "close": float(close)})

ctx = DataContext.from_adapters(
    InMemoryMarketAdapter(pd.DataFrame(rows)),
    calendars={"XNYS": cal},
    store=None,
)

features = [
    FeatureRequestGroup(
        key="volatility",
        tags={"recipe": "quickstart"},
        requests=[
            FeatureRequest(
                template=LagReturnsTemplate(),
                key="returns",
                params={
                    "dataset": "market.ohlcv",
                    "source": "market",
                    "price_col": "close",
                    "lags": [1, 5, 10],
                },
            ),
            FeatureRequest(
                template=RollingVolatilityTemplate(),
                key="trailing_vol",
                params={
                    "dataset": "market.ohlcv",
                    "source": "market",
                    "price_col": "close",
                    "windows": [5, 10],
                    "lag": 1,
                    "annualization_factor": 252,
                },
            ),
        ],
    )
]

spec = DatasetSpec(
    universe=UniverseSpec(entities=entities),
    time=TimeSpec(
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-02-16"),
        calendar="XNYS",
        grid="B",
    ),
    features=features,
    target=TargetRequest(
        template=NextDaySqLogRetTarget(),
        params={},
        horizon=1,
        name="y",
    ),
    join_policy=JoinPolicy(how="inner", sort_index=True),
    missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    name="demo_dataset",
)

artifact = build_dataset(ctx, spec, persist=False)
print(artifact.X.shape)
print(int(artifact.y.notna().sum()))
```

For a full runnable version of this pattern, see
`examples/volatility_dataset_recipe.py`.

See [Dataset Spec guide](../guides/dataset-spec.md) for the full contract and
[Research Recipes](../guides/research-recipes.md) for notebook-shaped patterns.
