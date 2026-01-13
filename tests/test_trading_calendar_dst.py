import pandas as pd
from alphaforge.time.calendar import TradingCalendar


def test_xnys_session_close_shifts_across_dst():
    # Winter (no DST): 2021-01-04 close should be 21:00 UTC
    cal = TradingCalendar("XNYS", tz="America/New_York")
    close_winter = cal.session_close_utc(pd.Timestamp("2021-01-04"))
    assert close_winter.tz is not None and close_winter.hour == 21

    # Summer (DST): 2021-07-06 close should be 20:00 UTC (16:00 ET -> 20:00 UTC)
    close_summer = cal.session_close_utc(pd.Timestamp("2021-07-06"))
    assert close_summer.tz is not None and close_summer.hour == 20

    # verify trading minutes exist and are tz-aware
    mins = cal.trading_minutes_utc(
        pd.Timestamp("2021-07-06T00:00:00Z"),
        pd.Timestamp("2021-07-06T23:59:59Z"),
        freq="15min",
    )
    assert isinstance(mins, pd.DatetimeIndex)
    assert mins.tz is not None
    # If trading minutes exist for the day, they should fall within a reasonable UTC window
    if len(mins) > 0:
        assert mins.min() >= pd.Timestamp("2021-07-06T12:00:00Z")
        assert mins.max() <= pd.Timestamp("2021-07-06T22:00:00Z")
