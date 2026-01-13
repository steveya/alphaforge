import pandas as pd
import numpy as np

from alphaforge.data.schema import TableSchema
from alphaforge.data.query import Query


class DummySource:
    name = "dummy"

    def __init__(self, ohlcv_long: pd.DataFrame, macro_long: pd.DataFrame):
        self._ohlcv = ohlcv_long.copy()
        self._macro = macro_long.copy()
        self._schemas = {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=["close"],
                canonical_columns=["close"],
                native_freq="B",
                expected_cadence_days=1,
            ),
            "macro.series": TableSchema(
                name="macro.series",
                required_columns=["value"],
                canonical_columns=["value"],
                native_freq="M",
                expected_cadence_days=30,
            ),
        }

    def schemas(self):
        return self._schemas

    def fetch(self, q: Query) -> pd.DataFrame:
        df = self._ohlcv if q.table == "market.ohlcv" else self._macro
        keep = ["date", "entity_id"] + list(q.columns)
        out = df[keep].copy()

        if q.start is not None:
            out = out[out["date"] >= pd.Timestamp(q.start)]
        if q.end is not None:
            out = out[out["date"] <= pd.Timestamp(q.end)]
        if q.entities is not None:
            out = out[out["entity_id"].isin([str(x) for x in q.entities])]

        return out.reset_index(drop=True)
