from __future__ import annotations

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
from alphaforge.features.template import SliceSpec
from alphaforge.time.calendar import TradingCalendar


class InMemoryMarketAdapter(SourceAdapterBase):
    source_name = "market"
    datasets = frozenset({"market.ohlcv"})

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        self.fetch_calls: list[Query] = []

    def fetch(self, query: Query, *, max_staleness=None) -> FetchResult:
        self.fetch_calls.append(query)
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


def _market_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    rows: list[dict[str, object]] = []
    values = {
        "AAA": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
        "BBB": [50.0, 49.0, 50.0, 51.0, 52.0, 53.0],
    }
    for entity, closes in values.items():
        for obs_date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "series_key": entity,
                    "obs_date": obs_date,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _ctx() -> tuple[DataContext, InMemoryMarketAdapter]:
    adapter = InMemoryMarketAdapter(_market_frame())
    ctx = DataContext.from_adapters(
        adapter,
        calendars={"XNYS": TradingCalendar("XNYS", tz="UTC")},
        store=None,
    )
    return ctx, adapter


def _price_index(frame: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> pd.Series:
    out = frame.copy()
    out["ts_utc"] = pd.to_datetime(out["obs_date"], utc=True)
    if asof is not None:
        out = out[out["ts_utc"] <= asof]
    return (
        out.set_index(["ts_utc", "series_key"])["close"]
        .rename_axis(index=["ts_utc", "entity_id"])
        .sort_index()
    )


def test_lag_returns_template_uses_adapter_load_and_local_asof_filter() -> None:
    ctx, adapter = _ctx()
    asof = pd.Timestamp("2024-01-08T00:00:00Z")
    template = LagReturnsTemplate()

    ff = template.transform(
        ctx,
        {
            "dataset": "market.ohlcv",
            "source": "market",
            "price_col": "close",
            "lags": [1, 2],
        },
        SliceSpec(
            start=pd.Timestamp("2024-01-02T00:00:00Z"),
            end=pd.Timestamp("2024-01-12T00:00:00Z"),
            entities=["AAA", "BBB"],
            asof=asof,
            grid="B",
        ),
        None,
    )

    assert len(adapter.fetch_calls) == 1
    assert adapter.fetch_calls[0].table == "market.ohlcv"
    assert set(ff.catalog["lag"]) == {1, 2}
    assert all(ff.X.index.get_level_values("ts_utc") <= asof)

    prices = _price_index(_market_frame(), asof=asof).astype(float)
    logret = np.log(prices).groupby(level="entity_id").diff()
    expected_lag_1 = logret.groupby(level="entity_id").shift(1).rename("expected")
    feature_id = ff.catalog[ff.catalog["lag"] == 1].index[0]
    got = ff.X[feature_id].rename("expected")
    pd.testing.assert_series_equal(got, expected_lag_1, check_names=True)


def test_rolling_volatility_template_builds_annualized_shifted_windows() -> None:
    ctx, _ = _ctx()
    template = RollingVolatilityTemplate()

    ff = template.transform(
        ctx,
        {
            "dataset": "market.ohlcv",
            "source": "market",
            "price_col": "close",
            "windows": [3],
            "lag": 1,
            "annualization_factor": 252,
        },
        SliceSpec(
            start=pd.Timestamp("2024-01-02T00:00:00Z"),
            end=pd.Timestamp("2024-01-12T00:00:00Z"),
            entities=["AAA", "BBB"],
            asof=None,
            grid="B",
        ),
        None,
    )

    prices = _price_index(_market_frame()).astype(float)
    logret = np.log(prices).groupby(level="entity_id").diff()
    expected = logret.groupby(level="entity_id").transform(
        lambda values: values.rolling(window=3, min_periods=3).std()
    )
    expected = expected.groupby(level="entity_id").shift(1) * np.sqrt(252.0)

    feature_id = ff.catalog[ff.catalog["window"] == 3].index[0]
    got = ff.X[feature_id].rename(expected.name)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_dataset_spec_recipe_can_use_built_in_market_templates() -> None:
    ctx, _ = _ctx()

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA", "BBB"]),
        time=TimeSpec(
            start=pd.Timestamp("2024-01-02T00:00:00Z"),
            end=pd.Timestamp("2024-01-12T00:00:00Z"),
            calendar="XNYS",
            grid="B",
        ),
        features=[
            FeatureRequestGroup(
                key="volatility",
                tags={"recipe": "volatility"},
                requests=[
                    FeatureRequest(
                        template=LagReturnsTemplate(),
                        key="returns",
                        params={
                            "dataset": "market.ohlcv",
                            "source": "market",
                            "price_col": "close",
                            "lags": [1, 2],
                        },
                    ),
                    FeatureRequest(
                        template=RollingVolatilityTemplate(),
                        key="trailing_vol",
                        params={
                            "dataset": "market.ohlcv",
                            "source": "market",
                            "price_col": "close",
                            "windows": [3, 5],
                            "lag": 1,
                            "annualization_factor": 252,
                        },
                    ),
                ],
            )
        ],
        target=TargetRequest(
            template=RollingVolatilityTemplate(),
            params={
                "dataset": "market.ohlcv",
                "source": "market",
                "price_col": "close",
                "windows": [3],
                "lag": 0,
                "annualization_factor": 252,
            },
            name="realized_vol_target",
        ),
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="keep"),
        name="volatility_recipe",
    )

    artifact = build_dataset(ctx, spec, persist=False)

    assert not artifact.X.empty
    assert artifact.X.shape[1] == 4
    assert artifact.y.name == "realized_vol_target"
    assert set(artifact.catalog["request_key"]) == {
        "volatility/returns",
        "volatility/trailing_vol",
    }
    assert set(artifact.catalog["template_name"]) == {
        "lag_returns",
        "rolling_volatility",
    }
