import pandas as pd
import pytest
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
)
from alphaforge.features.template import SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query


class SimpleSource:
    name = "simple"

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def schemas(self):
        from alphaforge.data.schema import TableSchema

        return {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=["close"],
                canonical_columns=["close"],
                native_freq="T",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        return self._df.reset_index(drop=True)


class BuggyTemplate:
    name = "buggy_raw_fetch"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        # Directly call source.fetch without passing asof (bad)
        src = ctx.sources["s"]
        df = src.fetch(
            Query(
                table="market.ohlcv",
                columns=["close"],
                start=slice.start,
                end=slice.end,
                entities=slice.entities,
                asof=None,
                grid=slice.grid,
            )
        )
        pf = pd.DataFrame(df).set_index(["date", "entity_id"]).sort_index()
        pf.index.set_names(["ts_utc", "entity_id"], inplace=True)
        X = pf[["close"]].rename(columns={"close": "f"})
        catalog = pd.DataFrame([{"feature_id": "f", "family": "buggy"}]).set_index(
            "feature_id"
        )
        return FeatureFrame(X=X, catalog=catalog, meta={})


def test_build_dataset_warns_on_template_leakage():
    times = [pd.Timestamp("2021-01-04T09:30:00Z"), pd.Timestamp("2021-01-04T09:35:00Z")]
    df = pd.DataFrame(
        {"date": times, "entity_id": ["AAA", "AAA"], "close": [10.0, 11.0]}
    )

    src = SimpleSource(df)
    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={"s": src}, calendars={"XNYS": cal}, store=None)

    asof = pd.Timestamp("2021-01-04T09:32:00Z")

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA"]),
        time=TimeSpec(
            start=pd.Timestamp("2021-01-04T00:00:00Z"),
            end=pd.Timestamp("2021-01-04T23:59:59Z"),
            calendar="XNYS",
            grid="5min",
            asof=asof,
        ),
        target=TargetRequest(template=BuggyTemplate()),
        features=[FeatureRequest(template=BuggyTemplate())],
    )

    with pytest.warns(UserWarning, match="returned .* rows with timestamps after asof"):
        art = build_dataset(ctx, spec, persist=False)
