"""Notebook-shaped volatility dataset recipe using canonical adapter loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphaforge import (  # noqa: E402
    DataContext,
    LagReturnsTemplate,
    RollingVolatilityTemplate,
)
from alphaforge.data.adapter import SourceAdapterBase  # noqa: E402
from alphaforge.data.query import Query  # noqa: E402
from alphaforge.data.types import FetchResult  # noqa: E402
from alphaforge.features.dataset_builder import build_dataset  # noqa: E402
from alphaforge.features.dataset_spec import (  # noqa: E402
    DatasetSpec,
    FeatureRequest,
    FeatureRequestGroup,
    JoinPolicy,
    MissingnessPolicy,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)
from alphaforge.features.target_template import TargetFrame  # noqa: E402
from alphaforge.features.template import SliceSpec  # noqa: E402
from alphaforge.time.calendar import TradingCalendar  # noqa: E402


class InMemoryMarketAdapter(SourceAdapterBase):
    source_name = "market"
    datasets = frozenset({"market.ohlcv"})

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def fetch(self, query: Query, *, max_staleness=None) -> FetchResult:
        frame = self._frame.copy()
        if query.entities is not None:
            frame = frame[frame["series_key"].isin(query.entities)]

        obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.start is not None:
            frame = frame[obs >= query.start]
            obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.end is not None:
            frame = frame[obs <= query.end]

        keep = ["series_key", "obs_date"] + [
            column for column in query.columns if column in frame.columns
        ]
        return FetchResult(
            data=frame[keep].reset_index(drop=True),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
        )

    def list_entities(self, dataset: str) -> list[str]:
        return sorted(self._frame["series_key"].unique())


class NextDaySquaredLogReturnTarget:
    name = "next_day_squared_log_return"
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
        target = (logret.groupby(level="entity_id").shift(-1) ** 2).rename("target")
        return TargetFrame(
            y=target,
            meta={"definition": "next-period squared log return"},
        )


def _market_frame() -> pd.DataFrame:
    calendar = TradingCalendar("XNYS", tz="UTC")
    dates = calendar.sessions("2024-01-02", "2024-02-16")
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for entity in ["AAA", "BBB"]:
        prices = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=len(dates)))
        for obs_date, close in zip(dates, prices, strict=True):
            rows.append(
                {
                    "series_key": entity,
                    "obs_date": obs_date,
                    "close": float(close),
                    "volume": float(rng.integers(900, 1100)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    calendar = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext.from_adapters(
        InMemoryMarketAdapter(_market_frame()),
        calendars={"XNYS": calendar},
        store=None,
    )

    features = [
        FeatureRequestGroup(
            key="volatility",
            tags={"recipe": "volatility", "asset_class": "equity"},
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
        universe=UniverseSpec(entities=["AAA", "BBB"]),
        time=TimeSpec(
            start=pd.Timestamp("2024-01-02T00:00:00Z"),
            end=pd.Timestamp("2024-02-16T00:00:00Z"),
            calendar="XNYS",
            grid="B",
        ),
        features=features,
        target=TargetRequest(
            template=NextDaySquaredLogReturnTarget(),
            horizon=1,
            name="next_day_sq_logret",
        ),
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
        name="volatility_recipe",
    )

    artifact = build_dataset(ctx, spec, persist=False)
    print("X shape:", artifact.X.shape)
    print("y non-null:", int(artifact.y.notna().sum()))
    print("Catalog columns:", sorted(artifact.catalog.columns.tolist()))
    print(artifact.catalog[["request_key", "template_name", "group_path"]].head())


if __name__ == "__main__":
    main()
