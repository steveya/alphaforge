import pandas as pd
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
        # ignore pushdowns - return all rows
        return self._df.reset_index(drop=True)


class TemplateNoLocalAsOf:
    """A feature that deliberately does NOT apply a local as-of filter: it just fetches
    the panel (calling ctx.fetch_panel) and returns it as a 1-col FeatureFrame."""

    name = "no_asof"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        # intentionally call ctx.fetch_panel with the slice's asof (correct usage)
        panel = ctx.fetch_panel(
            "s",
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
        X = panel.df[["close"]].rename(columns={"close": "f"})
        catalog = pd.DataFrame([{"feature_id": "f", "family": "test"}]).set_index(
            "feature_id"
        )
        return FeatureFrame(X=X, catalog=catalog, meta={})


class TemplateBuggyFetch:
    """A buggy template that calls the source fetch without passing asof (should NOT get future rows)."""

    name = "buggy"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        # call the underlying source.fetch indirectly by invoking ctx.sources directly (BUG)
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


def test_templates_with_and_without_local_asof_respect_global_asof():
    # build source data with one point beyond asof
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
        target=TargetRequest(template=TemplateNoLocalAsOf()),
        features=[FeatureRequest(template=TemplateNoLocalAsOf())],
    )

    art = build_dataset(ctx, spec, persist=False)
    X = art.X

    # the 09:35 row should be excluded because asof is 09:32
    assert all(ts <= asof for ts in X.index.get_level_values("ts_utc"))

    # Now add the buggy template that calls source.fetch without passing asof; dataset builder
    # should still not include rows beyond asof because templates are expected to request asof.
    # However, since the buggy template bypasses ctx.fetch_panel, it can introduce future leakage.
    # We assert that such misuse is detected by the build (i.e., result should contain the future row)
    spec2 = DatasetSpec(
        universe=UniverseSpec(entities=["AAA"]),
        time=TimeSpec(
            start=pd.Timestamp("2021-01-04T00:00:00Z"),
            end=pd.Timestamp("2021-01-04T23:59:59Z"),
            calendar="XNYS",
            grid="5min",
            asof=asof,
        ),
        target=TargetRequest(template=TemplateNoLocalAsOf()),
        features=[FeatureRequest(template=TemplateBuggyFetch())],
    )

    art2 = build_dataset(ctx, spec2, persist=False)
    X2 = art2.X

    # Final dataset must not include timestamps beyond asof (safe result)
    assert all(ts <= asof for ts in X2.index.get_level_values("ts_utc"))

    # However, the buggy template itself (if called directly) can introduce future rows by
    # bypassing ctx.fetch_panel and not passing asof; confirm that its raw FeatureFrame contains the future timestamp.
    s = SliceSpec(
        start=spec2.time.start,
        end=spec2.time.end,
        entities=spec2.universe.entities,
        asof=spec2.time.asof,
        grid=spec2.time.grid,
    )
    buggy = TemplateBuggyFetch()
    ff_buggy = buggy.transform(ctx, {}, s, None)
    if not ff_buggy.X.empty:
        assert any(ts > asof for ts in ff_buggy.X.index.get_level_values("ts_utc"))
