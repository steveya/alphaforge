from __future__ import annotations

import json
from typing import Tuple

import pandas as pd


def attach_importance(
    catalog: pd.DataFrame,
    importance: pd.Series,
    col: str = "importance",
) -> pd.DataFrame:
    """
    Join a feature-importance Series (index=feature_id) onto catalog (index=feature_id).
    """
    out = catalog.copy()
    # Ensure aligned index names
    if out.index.name is None:
        out.index.name = "feature_id"
    if importance.index.name is None:
        importance.index.name = out.index.name
    out[col] = importance.reindex(out.index).astype(float)
    return out


def data_importance(
    catalog_with_imp: pd.DataFrame,
    by: Tuple[str, ...] = ("source_table", "source_col"),
    col: str = "importance",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate importance by data fields (e.g., table/column).
    """
    return (
        catalog_with_imp.groupby(list(by), dropna=False)[col]
        .agg(agg)
        .sort_values(ascending=False)
        .to_frame(col)
    )


def tag_importance(
    catalog_with_imp: pd.DataFrame,
    tag_key: str,
    *,
    col: str = "importance",
    tags_col: str = "tags_json",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate importance by a single tag key. Supports dict or JSON-string tags.
    """

    def _extract(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, dict):
            return x.get(tag_key)
        if isinstance(x, str):
            try:
                return json.loads(x).get(tag_key)
            except Exception:
                return None
        return None

    tmp = catalog_with_imp.copy()
    tmp["_tag_value"] = tmp[tags_col].map(_extract)

    return (
        tmp.dropna(subset=["_tag_value"])
        .groupby("_tag_value", dropna=False)[col]
        .agg(agg)
        .sort_values(ascending=False)
        .to_frame(col)
        .rename_axis(tag_key)
    )
