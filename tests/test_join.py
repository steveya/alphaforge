import pandas as pd
import numpy as np
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ops import join_feature_frames


def test_join_feature_frames_catalog_and_index():
    idx1 = pd.MultiIndex.from_product(
        [[pd.Timestamp("2020-01-01")], ["AAA"]], names=["ts_utc", "entity_id"]
    )
    f1 = FeatureFrame(
        X=pd.DataFrame({"a": [1.0]}, index=idx1),
        catalog=pd.DataFrame([{"feature_id": "a", "family": "x"}]).set_index(
            "feature_id"
        ),
        meta={},
    )

    idx2 = pd.MultiIndex.from_product(
        [[pd.Timestamp("2020-01-01")], ["AAA"]], names=["ts_utc", "entity_id"]
    )
    f2 = FeatureFrame(
        X=pd.DataFrame({"b": [2.0]}, index=idx2),
        catalog=pd.DataFrame([{"feature_id": "b", "family": "y"}]).set_index(
            "feature_id"
        ),
        meta={},
    )

    j = join_feature_frames([f1, f2])
    assert list(j.X.columns) == ["a", "b"]
    assert "a" in j.catalog.index and "b" in j.catalog.index
