import pandas as pd
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    SliceOverride,
)
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.template import SliceSpec, ParamSpec
from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext
from alphaforge.features.dataset_spec import DatasetArtifact


class TinyTemplate:
    name = "tiny"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx, params, slice: SliceSpec, state):
        # intentionally create a sparse feature: only first session x first entity
        cal = ctx.calendars["XNYS"]
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        times = pd.DatetimeIndex([sessions[0]])
        idx = pd.MultiIndex.from_product(
            [times, [slice.entities[0]]], names=["ts_utc", "entity_id"]
        )
        X = pd.DataFrame({"f": [1.0]}, index=idx)
        catalog = pd.DataFrame([{"feature_id": "f", "family": "tiny"}]).set_index(
            "feature_id"
        )
        return FeatureFrame(X=X, catalog=catalog, meta={})


class DummyTarget:
    name = "dummy"
    version = "1.0"
    param_space = {}

    def transform(self, ctx, params, slice: SliceSpec, state):
        # return a series indexed at session dates only (no entity dim)
        cal = ctx.calendars["XNYS"]
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        y = pd.Series([0.1, 0.2], index=pd.DatetimeIndex(sessions[:2]), name="target")
        return y


def test_dataset_alignment():
    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA", "BBB"]),
        time=TimeSpec(
            start=pd.Timestamp("2021-01-04", tz="UTC"),
            end=pd.Timestamp("2021-01-05", tz="UTC"),
            calendar="XNYS",
            grid="B",
        ),
        target=TargetRequest(template=DummyTarget()),
        features=[FeatureRequest(template=TinyTemplate())],
    )

    from alphaforge.time.grids import build_grid_utc

    grid = build_grid_utc(cal, spec.time.start, spec.time.end, spec.time.grid)
    assert len(grid) == 2

    art: DatasetArtifact = build_dataset(ctx, spec, persist=False)
    X = art.X
    y = art.y

    # index should be MultiIndex (ts_utc x entity), final rows trimmed by missingness policy
    assert X.index.names == ["ts_utc", "entity_id"]
    # After drop_if_any_nan, only the cell where the tiny feature exists remains
    assert len(X) == 1
    assert len(y) == 1
    # the tiny feature is present only in the first session-AAA cell
    grid = build_grid_utc(cal, spec.time.start, spec.time.end, spec.time.grid)
    assert X.loc[(grid[0], "AAA")]["f"] == 1.0
