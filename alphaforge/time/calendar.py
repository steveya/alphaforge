from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    """
    Minimal business-day calendar.

    - tz is the calendar's *local* timezone (e.g. America/New_York for XNYS).
    - session labels are returned as tz-aware UTC instants at 00:00 UTC by default.
      (They are labels; open/close time helpers are provided below.)
    """

    name: str
    tz: str = "UTC"

    def sessions(self, start: str, end: str) -> pd.DatetimeIndex:
        # session labels as tz-aware UTC timestamps
        return pd.bdate_range(start=start, end=end, tz="UTC")

    def next_session(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC") + pd.tseries.offsets.BDay(1)

    def prev_session(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC") - pd.tseries.offsets.BDay(1)

    def session_open_utc(self, session_label: pd.Timestamp | str) -> pd.Timestamp:
        """Return session open as tz-aware UTC Timestamp.

        session_label may be a session label returned by sessions() (tz-aware UTC midnight)
        or a date-like string / Timestamp.
        """
        ts = pd.Timestamp(session_label)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        # convert to local timezone and take that date's 09:30 local
        local = ts.tz_convert(self.tz).normalize()
        open_local = local + pd.Timedelta(hours=9, minutes=30)
        return open_local.tz_convert("UTC")

    def session_close_utc(self, session_label: pd.Timestamp | str) -> pd.Timestamp:
        """Return session close as tz-aware UTC Timestamp (default 16:00 local -> UTC)."""
        ts = pd.Timestamp(session_label)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.tz).normalize()
        close_local = local + pd.Timedelta(hours=16)
        return close_local.tz_convert("UTC")

    def trading_minutes_utc(
        self, start_utc: pd.Timestamp, end_utc: pd.Timestamp, freq: str = "5min"
    ) -> pd.DatetimeIndex:
        """Return tz-aware UTC DatetimeIndex of trading minutes between start_utc and end_utc.

        Generates minutes during each trading session (09:30..16:00 local) with frequency `freq`.
        """
        s = pd.Timestamp(start_utc)
        e = pd.Timestamp(end_utc)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")

        # session labels between s and e (based on UTC session labels)
        sessions = self.sessions(str(s.date()), str(e.date()))

        pieces = []
        for sess in sessions:
            open_utc = self.session_open_utc(sess)
            close_utc = self.session_close_utc(sess)
            # generate local range to respect DST boundaries then convert to UTC
            open_local = open_utc.tz_convert(self.tz)
            close_local = close_utc.tz_convert(self.tz)
            # start/end are tz-aware local timestamps; don't pass tz param to avoid pandas assertion
            rng_local = pd.date_range(
                start=open_local, end=close_local, freq=freq, inclusive="right"
            )
            if len(rng_local) == 0:
                continue
            rng_utc = rng_local.tz_convert("UTC")
            # filter by overall start/end
            rng_utc = rng_utc[(rng_utc >= s) & (rng_utc <= e)]
            pieces.append(rng_utc)

        if pieces:
            if len(pieces) == 1:
                out = pieces[0]
            else:
                out = pieces[0].append(pieces[1:])
                out = out.sort_values()
                out = pd.DatetimeIndex(out.unique()).sort_values()
            return out
        else:
            return pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
