from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.diagnostics.importance import attach_importance, data_importance, tag_importance
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
from alphaforge.pit import PITAccessor
from alphaforge.pit.transforms import PITTransformSpec
from alphaforge.time.calendar import TradingCalendar


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


def run_example(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    pit = PITAccessor.open(root_path)
    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDP", "GDP", "ALT", "ALT", "ALT"],
                "obs_date": [
                    pd.Timestamp("2024-01-31"),
                    pd.Timestamp("2024-03-31"),
                    pd.Timestamp("2024-01-31"),
                    pd.Timestamp("2024-02-29"),
                    pd.Timestamp("2024-03-31"),
                ],
                "asof_utc": [
                    pd.Timestamp("2024-04-10", tz="UTC"),
                    pd.Timestamp("2024-04-10", tz="UTC"),
                    pd.Timestamp("2024-04-05", tz="UTC"),
                    pd.Timestamp("2024-04-05", tz="UTC"),
                    pd.Timestamp("2024-04-05", tz="UTC"),
                ],
                "value": [3.0, 4.0, 1.0, 2.0, 3.0],
                "source": ["docs-example"] * 5,
            }
        )
    )

    pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP",
            output_series_key="GDP_spliced",
            op="coalesce",
            params={"other_series_keys": ["ALT"]},
        ),
        overwrite=True,
    )
    pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP_spliced",
            output_series_key="GDP_spliced_pct",
            op="pct_change",
            params={"periods": 1},
        ),
        overwrite=True,
    )
    pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP_spliced",
            output_series_key="GDP_spliced_count",
            op="aggregate",
            params={"agg": "count"},
        ),
        overwrite=True,
    )

    asof = pd.Timestamp("2024-04-10", tz="UTC")
    splice_snapshot = pit.get_snapshot("GDP_spliced", asof)
    pct_change_snapshot = pit.get_snapshot("GDP_spliced_pct", asof)
    count_snapshot = pit.get_snapshot("GDP_spliced_count", asof)

    lineage_row = pit.conn.execute(
        """
        SELECT meta_json
        FROM pit_observations
        WHERE series_key = ? AND obs_date = ? AND asof_utc = ?
        """,
        [
            "GDP_spliced",
            pd.Timestamp("2024-02-29"),
            pd.Timestamp("2024-04-10"),
        ],
    ).fetchone()
    lineage = json.loads(str(lineage_row[0])) if lineage_row is not None and lineage_row[0] else {}

    cal = TradingCalendar("XNYS", tz="UTC")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)
    dataset = build_dataset(
        ctx,
        DatasetSpec(
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
                        template_tags={"stage": "template"},
                    ),
                    tags={"group": "raw", "stage": "request"},
                ),
                FeatureRequest(
                    template=_TaggedFeatureTemplate(
                        feature_id="feat_abs",
                        values=[3.0, 4.0],
                        source_col="abs_close",
                    ),
                    tags={"group": "abs"},
                ),
            ],
        ),
        persist=False,
    )
    importance = pd.Series({"feat_raw": 0.7, "feat_abs": 0.3}, name="importance")
    importance.index.name = "feature_id"
    catalog_with_importance = attach_importance(dataset.catalog, importance)

    return {
        "splice_snapshot": splice_snapshot,
        "pct_change_snapshot": pct_change_snapshot,
        "count_snapshot": count_snapshot,
        "tag_rollup": tag_importance(catalog_with_importance, tag_key="group"),
        "data_rollup": data_importance(catalog_with_importance),
        "selected_fallback_series_key": lineage.get("selected_input_series_key"),
        "selected_fallback_asof": lineage.get("selected_input_asof_utc"),
    }
