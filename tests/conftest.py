import pandas as pd
import numpy as np
import pytest

from alphaforge.data.schema import TableSchema
from alphaforge.data.query import Query
from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext


class MemoryStore:
    """Fast in-memory Store for tests."""

    def __init__(self):
        self.frames = {}
        self.states = {}

    def get_frame(self, realization_id):
        return self.frames.get(realization_id)

    def put_frame(self, realization_id, frame):
        self.frames[realization_id] = frame

    def exists_frame(self, realization_id):
        return realization_id in self.frames

    def put_state(self, state, payload: bytes):
        # store payload directly
        self.states[state.state_id] = payload
        # minimal Artifact object contract
        from alphaforge.features.frame import Artifact

        return Artifact(
            artifact_id=state.state_id,
            kind="fit_state",
            uri=f"memory://{state.state_id}",
            meta=state.meta,
        )

    def get_state(self, state_id):
        return self.states[state_id]


class DummySource:
    """In-memory DataSource supporting one daily OHLCV table and one monthly macro table."""

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
                # In MVP we won't use release_time_column; later you will.
            ),
        }

    def schemas(self):
        return self._schemas

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table == "market.ohlcv":
            df = self._ohlcv
        elif q.table == "macro.series":
            df = self._macro
        else:
            raise KeyError(q.table)

        # column projection
        keep = ["date", "entity_id"] + list(q.columns)
        df = df[keep].copy()

        # coerce df['date'] to tz-aware UTC for robust comparisons
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")

        # time filter
        if q.start is not None:
            start_ts = pd.Timestamp(q.start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            df = df[df["date"] >= start_ts]
        if q.end is not None:
            end_ts = pd.Timestamp(q.end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            df = df[df["date"] <= end_ts]

        # entity filter
        if q.entities is not None:
            df = df[df["entity_id"].isin([str(x) for x in q.entities])]
        # asof/vintage ignored in MVP dummy
        return df.reset_index(drop=True)


@pytest.fixture
def dummy_ctx():
    cal = TradingCalendar("XNYS", tz="UTC")
    dates = cal.sessions("2020-01-01", "2020-03-31").tz_convert(None)

    # two assets
    entities = ["AAA", "BBB"]

    # daily close: simple random walk
    rows = []
    rng = np.random.default_rng(0)
    for e in entities:
        px = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
        for d, p in zip(dates, px):
            rows.append({"date": d, "entity_id": e, "close": float(p)})
    ohlcv = pd.DataFrame(rows)

    # monthly macro: value only on month end (structural missingness)
    month_ends = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    macro_rows = []
    for d, v in zip(month_ends, [1.0, 2.0, 3.0]):
        macro_rows.append({"date": d, "entity_id": "CPI", "value": float(v)})
    macro = pd.DataFrame(macro_rows)

    store = MemoryStore()
    src = DummySource(ohlcv_long=ohlcv, macro_long=macro)
    ctx = DataContext(
        sources={"dummy": src},
        calendars={"XNYS": cal},
        store=store,
        universe=None,
        entity_meta=None,
    )
    return ctx, dates, entities
