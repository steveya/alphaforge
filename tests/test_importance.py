from __future__ import annotations

import json
import pandas as pd
import numpy as np

from alphaforge.diagnostics.importance import (
    attach_importance,
    data_importance,
    tag_importance,
)


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
