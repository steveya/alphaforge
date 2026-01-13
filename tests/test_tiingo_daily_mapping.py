import pandas as pd
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.time.calendar import TradingCalendar


class DummyDailySource:
    name = "dummy_daily"

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def schemas(self):
        from alphaforge.data.schema import TableSchema

        return {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=["close"],
                canonical_columns=["close"],
                native_freq="B",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        # return as naive dates at midnight UTC
        return self._df.reset_index(drop=True)


def test_daily_date_to_session_close_mapping():
    # dates are date labels (midnight)
    dates = [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")]
    df = pd.DataFrame(
        {"date": dates, "entity_id": ["AAA", "AAA"], "close": [100.0, 101.0]}
    )

    src = DummyDailySource(df)
    cal = TradingCalendar("XNYS", tz="America/New_York")
    ctx = DataContext(sources={"d": src}, calendars={"XNYS": cal}, store=None)

    q = Query(table="market.ohlcv", columns=["close"], grid="sessions:XNYS")
    panel = ctx.fetch_panel("d", q)

    times = pd.DatetimeIndex(panel.df.index.get_level_values("ts_utc"))
    # session close for 2021-01-04 is 2021-01-04 21:00 UTC
    assert times[0].hour == 21 and times[0].tz is not None
    assert times[1] > times[0]
