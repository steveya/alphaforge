import json

import pandas as pd
import pytest

from alphaforge.data.context import DataContext
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    FeatureRequest,
    FeatureRequestGroup,
    JoinPolicy,
    MissingnessPolicy,
    SliceOverride,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.template import SliceSpec
from alphaforge.time.calendar import TradingCalendar


class _RecordingTemplate:
    version = "1.0"
    param_space = {}

    def __init__(self, name: str):
        self.name = name
        self.seen_slices: list[SliceSpec] = []

    def requires(self, params):
        return []

    def transform(self, ctx, params, slice: SliceSpec, state):
        del params, state
        self.seen_slices.append(slice)
        cal = ctx.calendars["XNYS"]
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        idx = pd.MultiIndex.from_product(
            [pd.DatetimeIndex([sessions[0]]), [slice.entities[0]]],
            names=["ts_utc", "entity_id"],
        )
        X = pd.DataFrame({self.name: [1.0]}, index=idx)
        catalog = pd.DataFrame(
            [{"feature_id": self.name, "family": "test"}]
        ).set_index("feature_id")
        return FeatureFrame(X=X, catalog=catalog, meta={})


class _TinyTarget:
    name = "target"
    version = "1.0"
    param_space = {}

    def transform(self, ctx, params, slice: SliceSpec, state):
        del params, state
        cal = ctx.calendars["XNYS"]
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        return pd.Series(
            [0.1] * len(sessions),
            index=pd.DatetimeIndex(sessions),
            name="target",
        )


def test_feature_request_group_composes_keys_tags_and_slice_overrides():
    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)

    gdp = _RecordingTemplate("gdp_level")
    cpi = _RecordingTemplate("cpi_level")
    nested_asof = pd.Timestamp("2024-01-03T16:00:00Z")

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA"]),
        time=TimeSpec(
            start=pd.Timestamp("2024-01-08T00:00:00Z"),
            end=pd.Timestamp("2024-01-10T00:00:00Z"),
            calendar="XNYS",
            grid="B",
            asof=pd.Timestamp("2024-01-10T16:00:00Z"),
        ),
        target=TargetRequest(template=_TinyTarget()),
        features=[
            FeatureRequestGroup(
                key="macro",
                tags={"family": "macro", "stage": "group"},
                slice_override=SliceOverride(lookback=pd.Timedelta(days=7)),
                requests=[
                    FeatureRequest(
                        template=gdp,
                        key="gdp",
                        tags={"series": "gdp"},
                    ),
                    FeatureRequestGroup(
                        key="inflation",
                        tags={"stage": "nested"},
                        requests=[
                            FeatureRequest(
                                template=cpi,
                                key="cpi",
                                tags={"series": "cpi"},
                                slice_override=SliceOverride(asof=nested_asof),
                            )
                        ],
                    ),
                ],
            )
        ],
        missingness=MissingnessPolicy(final_row_policy="keep"),
    )

    artifact = build_dataset(ctx, spec, persist=False)
    catalog = artifact.catalog

    assert gdp.seen_slices[0].start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert gdp.seen_slices[0].asof == pd.Timestamp("2024-01-10T16:00:00Z")
    assert cpi.seen_slices[0].start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert cpi.seen_slices[0].asof == nested_asof

    assert catalog.loc["gdp_level", "request_key"] == "macro/gdp"
    assert catalog.loc["cpi_level", "request_key"] == "macro/inflation/cpi"

    assert json.loads(catalog.loc["gdp_level", "tags_json"]) == {
        "family": "macro",
        "stage": "group",
        "series": "gdp",
    }
    assert json.loads(catalog.loc["cpi_level", "tags_json"]) == {
        "family": "macro",
        "stage": "nested",
        "series": "cpi",
    }


def test_dataset_contract_validates_join_and_missingness_policies():
    with pytest.raises(ValueError, match="JoinPolicy.how"):
        JoinPolicy(how="left")

    with pytest.raises(ValueError, match="MissingnessPolicy.final_row_policy"):
        MissingnessPolicy(final_row_policy="drop_some")

