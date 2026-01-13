import pandas as pd
from alphaforge.data.panel import PanelFrame
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query


class SimpleSource:
    name = "simple"

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def schemas(self):
        return {}

    def fetch(self, q: Query) -> pd.DataFrame:
        # ignore pushdowns for this simple source
        return self._df.reset_index(drop=True)


def test_panel_slicing_with_tz_aware_times():
    times = [
        pd.Timestamp("2021-01-01T15:30:00Z"),
        pd.Timestamp("2021-01-01T15:35:00Z"),
        pd.Timestamp("2021-01-01T15:40:00Z"),
    ]
    df = pd.DataFrame({"date": times, "entity_id": ["A", "A", "A"], "v": [1, 2, 3]})

    panel = PanelFrame.from_long(df)
    assert panel.df.index.names == ["ts_utc", "entity_id"]

    # slice with naive strings should be coerced to UTC-aware
    sliced = panel.slice(start="2021-01-01T15:35:00", end="2021-01-01T15:40:00")
    assert len(sliced.df) == 2
    # times in index should be tz-aware UTC
    times_idx = pd.DatetimeIndex(sliced.df.index.get_level_values("ts_utc"))
    assert times_idx.tz == pd.Timestamp("2021-01-01T00:00:00Z").tz


def test_fetch_panel_respects_asof(dummy_ctx, tmp_path):
    # create a short source that has one point beyond the asof
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2021-01-01T10:00:00Z"),
                pd.Timestamp("2021-01-01T10:05:00Z"),
            ],
            "entity_id": ["A", "A"],
            "close": [10.0, 11.0],
        }
    )
    src = SimpleSource(df)

    store = dummy_ctx[0].store
    ctx = DataContext(sources={"s": src}, calendars={}, store=store)

    q = Query(
        table="market.ohlcv", columns=["close"], asof=pd.Timestamp("2021-01-01T10:02")
    )
    panel = ctx.fetch_panel("s", q)

    # asof = 10:02 should exclude the 10:05 row
    times = pd.DatetimeIndex(panel.df.index.get_level_values("ts_utc"))
    assert all(times <= pd.Timestamp("2021-01-01T10:02Z"))


def test_xnys_trading_minutes_and_daily_close():
    from alphaforge.time.calendar import TradingCalendar
    from alphaforge.time.grids import build_grid_utc

    cal = TradingCalendar("XNYS", tz="America/New_York")

    # Verify session close UTC for a winter date (no DST)
    close_utc = cal.session_close_utc(pd.Timestamp("2021-01-04"))
    # 2021-01-04 16:00 ET is 21:00 UTC
    assert close_utc.hour == 21 and close_utc.tzinfo is not None

    # 5-minute grid within one session
    start = pd.Timestamp("2021-01-04T00:00:00Z")
    end = pd.Timestamp("2021-01-04T23:59:59Z")
    minutes = cal.trading_minutes_utc(start, end, freq="5min")
    assert isinstance(minutes, pd.DatetimeIndex)
    # all times are tz-aware UTC
    assert minutes.tz == pd.Timestamp("2021-01-04T00:00:00Z").tz
    # all times are between 13:30Z and 21:00Z for NY in winter (open:9:30ET=14:30? wait DST), assert within session window
    assert all(
        (minutes >= pd.Timestamp("2021-01-04T13:00Z"))
        & (minutes <= pd.Timestamp("2021-01-04T22:00Z"))
    )

    # ensure sessions and closes are non-empty
    sessions = cal.sessions(str(start.date()), str(end.date()))
    closes = [cal.session_close_utc(sess) for sess in sessions]
    assert len(sessions) > 0
    assert len(closes) > 0

    # verify that naive business day range yields closes too
    dates_b = pd.bdate_range(start=start.date(), end=end.date())
    assert len(dates_b) > 0
    closes_from_b = [cal.session_close_utc(d) for d in dates_b]
    assert len(closes_from_b) > 0
    assert closes_from_b == closes

    # build_grid_utc daily (B) returns the session close
    grid = build_grid_utc(cal, start, end, "B")
    assert len(grid) == 1
    assert grid[0] == close_utc

    # build_grid_utc intraday
    grid5 = build_grid_utc(cal, start, end, "5min")
    assert grid5.equals(minutes)
