import pandas as pd
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
)
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.template import SliceSpec
from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext


class TinyIntradayTemplate:
    name = "tiny_intraday"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        cal = ctx.calendars["XNYS"]
        minutes = cal.trading_minutes_utc(slice.start, slice.end, freq="5min")
        # choose the first two minutes
        chosen = minutes[:2]
        if slice.asof is not None:
            chosen = chosen[chosen <= slice.asof]
        if len(chosen) == 0:
            # return empty frame
            return FeatureFrame(
                X=pd.DataFrame(columns=["f"]), catalog=pd.DataFrame(), meta={}
            )

        idx = pd.MultiIndex.from_product(
            [chosen, [slice.entities[0]]], names=["ts_utc", "entity_id"]
        )
        X = pd.DataFrame({"f": [1.0] * len(chosen)}, index=idx)
        catalog = pd.DataFrame(
            [{"feature_id": "f", "family": "tiny_intraday"}]
        ).set_index("feature_id")
        return FeatureFrame(X=X, catalog=catalog, meta={})


class DummyIntradayTarget:
    name = "dummy_intraday"
    version = "1.0"
    param_space = {}

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        cal = ctx.calendars["XNYS"]
        minutes = cal.trading_minutes_utc(slice.start, slice.end, freq="5min")
        chosen = minutes[:2]
        if slice.asof is not None:
            chosen = chosen[chosen <= slice.asof]
        if len(chosen) == 0:
            # return empty series
            return pd.Series(dtype=float, name="target")
        # build a multiindex target (ts_utc x entity_id) so alignment is explicit
        idx = pd.MultiIndex.from_product(
            [chosen, [slice.entities[0], slice.entities[1]]],
            names=["ts_utc", "entity_id"],
        )
        X = pd.DataFrame({"target": [0.1] * len(idx)}, index=idx)
        # return as 1-col FeatureFrame which will be converted to Series by builder
        return FeatureFrame(
            X=X,
            catalog=pd.DataFrame(
                [{"feature_id": "target", "family": "target"}]
            ).set_index("feature_id"),
            meta={},
        )


def test_intraday_dataset_alignment_and_asof():
    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)

    # no asof: both minutes included
    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA", "BBB"]),
        time=TimeSpec(
            start=pd.Timestamp("2021-01-04T00:00:00Z"),
            end=pd.Timestamp("2021-01-04T23:59:59Z"),
            calendar="XNYS",
            grid="5min",
            asof=None,
        ),
        target=TargetRequest(template=DummyIntradayTarget()),
        features=[FeatureRequest(template=TinyIntradayTemplate())],
    )

    art = build_dataset(ctx, spec, persist=False)
    X = art.X
    y = art.y

    # grid should include at least 2 minutes
    grid = cal.trading_minutes_utc(spec.time.start, spec.time.end, freq="5min")
    assert len(grid) >= 2

    # Since TinyIntradayTemplate only emits the first two minutes for AAA, after drop_if_any_nan only rows with values remain
    assert X.index.names == ["ts_utc", "entity_id"]
    assert len(X) >= 1
    # verify that at least one AAA minute exists and has feature f
    assert any((idx[1] == "AAA" for idx in X.index))

    # now set asof to be between the two minutes so second minute is excluded
    m1 = grid[0]
    m2 = grid[1]
    mid_asof = m1 + (m2 - m1) / 2

    spec2 = DatasetSpec(
        universe=UniverseSpec(entities=["AAA", "BBB"]),
        time=TimeSpec(
            start=pd.Timestamp("2021-01-04T00:00:00Z"),
            end=pd.Timestamp("2021-01-04T23:59:59Z"),
            calendar="XNYS",
            grid="5min",
            asof=mid_asof,
        ),
        target=TargetRequest(template=DummyIntradayTarget()),
        features=[FeatureRequest(template=TinyIntradayTemplate())],
    )

    art2 = build_dataset(ctx, spec2, persist=False)
    X2 = art2.X

    # all feature times must be <= asof
    times = pd.DatetimeIndex(X2.index.get_level_values("ts_utc"))
    assert all(times <= mid_asof)
