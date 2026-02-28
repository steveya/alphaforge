from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from alphaforge.features.frame import FeatureFrame
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

_PARQUET_ENGINE_AVAILABLE = (
    find_spec("pyarrow") is not None or find_spec("fastparquet") is not None
)

pytestmark = pytest.mark.skipif(
    not _PARQUET_ENGINE_AVAILABLE,
    reason="pyarrow or fastparquet is required for parquet roundtrip tests",
)


def test_duckdb_store_roundtrip(tmp_path):
    store = DuckDBParquetStore(root=str(tmp_path))

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"), ["AAA"]],
        names=["ts_utc", "entity_id"],
    )
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    catalog = pd.DataFrame([{"feature_id": "f", "family": "test"}]).set_index(
        "feature_id"
    )
    frame = FeatureFrame(X=X, catalog=catalog, meta={"k": "v"})

    rid = "lag_returns:1.0:abc"
    store.put_frame(rid, frame)

    got = store.get_frame(rid)
    assert got is not None
    pd.testing.assert_frame_equal(got.X, frame.X, check_freq=False, check_exact=True)
    assert got.catalog.equals(frame.catalog)
    assert got.meta["k"] == "v"
