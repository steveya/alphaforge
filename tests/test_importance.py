from __future__ import annotations

import json

import numpy as np
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.diagnostics.importance import (
    attach_importance,
    data_importance,
    tag_importance,
)
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    FeatureRequest,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.template import SliceSpec
from alphaforge.time.calendar import TradingCalendar


def test_importance_aggregation_by_data_and_tags():
    # Build a tiny catalog with two features, each tagged differently
    cat = pd.DataFrame(
        [
            {
                "feature_id": "feat_raw_1",
                "source_table": "market.ohlcv",
                "source_col": "close",
                "family": "lag",
                "transform": "logret",
                "tags_json": json.dumps(
                    {"group": "raw", "stage": "return"}, sort_keys=True
                ),
            },
            {
                "feature_id": "feat_abs_1",
                "source_table": "market.ohlcv",
                "source_col": "close",
                "family": "lag",
                "transform": "abslogret",
                "tags_json": json.dumps(
                    {"group": "abs", "stage": "return"}, sort_keys=True
                ),
            },
        ]
    ).set_index("feature_id")

    # Dummy importance (indexed by feature_id)
    imp = pd.Series({"feat_raw_1": 0.7, "feat_abs_1": 0.3}, name="importance")
    imp.index.name = "feature_id"

    # Attach importance to catalog
    cat_imp = attach_importance(cat, imp, col="importance")
    assert np.isclose(cat_imp["importance"].sum(), 1.0)

    # Aggregate by data (table/column)
    di = data_importance(cat_imp, by=("source_table", "source_col"), col="importance")
    # Only one source table/col here; total equals 1.0
    assert np.isclose(di["importance"].sum(), 1.0)

    # Aggregate by tag group (raw vs abs)
    ti = tag_importance(
        cat_imp, tag_key="group", col="importance", tags_col="tags_json"
    )
    # Expect raw=0.7, abs=0.3
    assert np.isclose(ti.loc["raw", "importance"], 0.7)
    assert np.isclose(ti.loc["abs", "importance"], 0.3)


class _TaggedFeatureTemplate:
    name = "tagged"
    version = "1.0"
    param_space = {}

    def __init__(
        self,
        *,
        feature_id: str,
        values: list[float],
        source_col: str,
        template_tags: dict[str, str] | None = None,
    ):
        self.feature_id = feature_id
        self.values = values
        self.source_col = source_col
        self.template_tags = template_tags or {}

    def requires(self, params):
        return []

    def transform(self, ctx, params, slice: SliceSpec, state):
        del params, state
        cal = ctx.calendars["XNYS"]
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        idx = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(sessions), [slice.entities[0]]],
            names=["ts_utc", "entity_id"],
        )
        X = pd.DataFrame({self.feature_id: self.values[: len(idx)]}, index=idx)
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": self.feature_id,
                    "source_table": "macro.release",
                    "source_col": self.source_col,
                    "tags_json": json.dumps(self.template_tags, sort_keys=True),
                }
            ]
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


def test_dataset_build_stamps_request_tags_and_supports_importance_rollups():
    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA"]),
        time=TimeSpec(
            start=pd.Timestamp("2024-01-02", tz="UTC"),
            end=pd.Timestamp("2024-01-03", tz="UTC"),
            calendar="XNYS",
            grid="B",
        ),
        target=TargetRequest(template=_TinyTarget()),
        features=[
            FeatureRequest(
                template=_TaggedFeatureTemplate(
                    feature_id="feat_raw",
                    values=[1.0, 2.0],
                    source_col="close",
                    template_tags={"stage": "template", "template_only": "keep"},
                ),
                tags={"group": "raw", "stage": "request"},
            ),
            FeatureRequest(
                template=_TaggedFeatureTemplate(
                    feature_id="feat_abs",
                    values=[3.0, 4.0],
                    source_col="abs_close",
                    template_tags={"template_only": "keep"},
                ),
                tags={"group": "abs"},
            ),
        ],
    )

    artifact = build_dataset(ctx, spec, persist=False)
    catalog = artifact.catalog.copy()

    raw_tags = json.loads(catalog.loc["feat_raw", "tags_json"])
    abs_tags = json.loads(catalog.loc["feat_abs", "tags_json"])
    assert raw_tags == {
        "group": "raw",
        "stage": "request",
        "template_only": "keep",
    }
    assert abs_tags == {"group": "abs", "template_only": "keep"}

    importance = pd.Series({"feat_raw": 0.7, "feat_abs": 0.3}, name="importance")
    importance.index.name = "feature_id"
    catalog_with_importance = attach_importance(catalog, importance)

    by_data = data_importance(
        catalog_with_importance,
        by=("source_table", "source_col"),
        col="importance",
    )
    by_group = tag_importance(
        catalog_with_importance,
        tag_key="group",
        col="importance",
        tags_col="tags_json",
    )
    by_stage = tag_importance(
        catalog_with_importance,
        tag_key="stage",
        col="importance",
        tags_col="tags_json",
    )

    assert np.isclose(by_data.loc[("macro.release", "close"), "importance"], 0.7)
    assert np.isclose(by_data.loc[("macro.release", "abs_close"), "importance"], 0.3)
    assert np.isclose(by_group.loc["raw", "importance"], 0.7)
    assert np.isclose(by_group.loc["abs", "importance"], 0.3)
    assert np.isclose(by_stage.loc["request", "importance"], 0.7)
