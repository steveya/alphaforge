import numpy as np
import pandas as pd

from alphaforge.features.frame import FeatureFrame
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def test_duckdb_store_end_to_end(tmp_path):
    store = DuckDBParquetStore(root=str(tmp_path))

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC"), ["AAA", "BBB"]],
        names=["ts_utc", "entity_id"],
    )
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    catalog = pd.DataFrame([{"feature_id": "f", "family": "test"}]).set_index(
        "feature_id"
    )
    frame = FeatureFrame(X=X, catalog=catalog, meta={"source": "end_to_end"})

    store.put_frame("demo:1.0:xyz", frame)
    got = store.get_frame("demo:1.0:xyz")

    assert got is not None
    pd.testing.assert_frame_equal(got.X, frame.X, check_freq=False)
    pd.testing.assert_frame_equal(got.catalog, frame.catalog)
    assert got.meta["source"] == "end_to_end"
