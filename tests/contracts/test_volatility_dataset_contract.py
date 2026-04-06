from __future__ import annotations

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
from alphaforge.time.calendar import TradingCalendar


class _InMemoryMarketAdapter(SourceAdapterBase):
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


def _market_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC")
    rows = []
    for entity, closes in {
        "AAA": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 106.0, 108.0],
        "BBB": [50.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 56.0],
    }.items():
        for obs_date, close in zip(dates, closes, strict=True):
            rows.append({"series_key": entity, "obs_date": obs_date, "close": close})
    return pd.DataFrame(rows)


def test_volatility_recipe_contract_uses_canonical_short_path() -> None:
    ctx = DataContext.from_adapters(
        _InMemoryMarketAdapter(_market_frame()),
        calendars={"XNYS": TradingCalendar("XNYS", tz="UTC")},
        store=None,
    )

    loaded = ctx.load("market.ohlcv", columns=["close"], entities=["AAA"])
    assert loaded.source == "market"
    assert loaded.dataset == "market.ohlcv"

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
                            "lags": [1, 2],
                        },
                    ),
                    FeatureRequest(
                        template=RollingVolatilityTemplate(),
                        key="trailing_vol",
                        params={
                            "dataset": "market.ohlcv",
                            "source": "market",
                            "windows": [3],
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
                "windows": [3],
                "lag": 0,
                "annualization_factor": 252,
            },
            name="realized_vol_target",
        ),
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="keep"),
        name="volatility_contract",
    )

    artifact = build_dataset(ctx, spec, persist=False)

    assert not artifact.X.empty
    assert set(artifact.catalog["request_key"]) == {
        "volatility/returns",
        "volatility/trailing_vol",
    }
    assert set(artifact.catalog["template_name"]) == {
        "lag_returns",
        "rolling_volatility",
    }

