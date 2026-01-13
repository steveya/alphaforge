import pandas as pd
import numpy as np
from pathlib import Path
from alphaforge.store.local_parquet import LocalParquetStore
from alphaforge.features.frame import FeatureFrame


def test_local_store_roundtrip(tmp_path: Path):
    store = LocalParquetStore(str(tmp_path / "store"))

    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")], ["AAA", "BBB"]],
        names=["ts_utc", "entity_id"],
    )
    X = pd.DataFrame({"f": np.arange(len(idx))}, index=idx)
    catalog = pd.DataFrame([{"feature_id": "f", "family": "test"}]).set_index(
        "feature_id"
    )
    frame = FeatureFrame(X=X, catalog=catalog, meta={"hello": "world"})

    rid = "test:1.0:abcd"
    store.put_frame(rid, frame)

    got = store.get_frame(rid)
    assert got is not None
    assert got.X.equals(frame.X)
    assert got.catalog.equals(frame.catalog)
    assert got.meta["hello"] == "world"
