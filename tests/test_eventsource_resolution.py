import pandas as pd
import pytest

from alphaforge.time.events import SessionCloseEvents, FixedIntervalEvents


class DummyCtx:
    def __init__(self):
        # Minimal calendar registry for events.py to look up
        self.calendars = {"XNYS": object()}


def test_session_close_events_uses_build_grid_utc(monkeypatch):
    calls = []

    # Monkeypatch the function imported and used inside events.py
    def fake_build_grid_utc(cal, start_utc, end_utc, grid):
        calls.append((cal, start_utc, end_utc, grid))
        # Deterministic daily grid between start and end, UTC
        return pd.date_range(
            start=start_utc.normalize(), end=end_utc.normalize(), freq="D", tz="UTC"
        )

    monkeypatch.setattr(
        "alphaforge.time.events.build_grid_utc", fake_build_grid_utc, raising=True
    )

    ctx = DummyCtx()
    start = pd.Timestamp("2020-01-02 12:34:56Z")
    end = pd.Timestamp("2020-01-05 08:00:00Z")

    src = SessionCloseEvents(calendar="XNYS", delay=pd.Timedelta(0))
    out = src.events(ctx, start, end)

    # Assert build_grid_utc called with "sessions"
    assert len(calls) == 1
    _, s_called, e_called, grid_called = calls[0]
    assert s_called == start
    assert e_called == end
    assert grid_called == "sessions"

    # Returned timestamps equal base grid (no delay)
    expected = pd.date_range(
        start=start.normalize(), end=end.normalize(), freq="D", tz="UTC"
    )
    pd.testing.assert_index_equal(out, expected)


def test_session_close_events_applies_delay(monkeypatch):
    def fake_build_grid_utc(cal, start_utc, end_utc, grid):
        return pd.date_range(
            start=start_utc.normalize(), end=end_utc.normalize(), freq="D", tz="UTC"
        )

    monkeypatch.setattr(
        "alphaforge.time.events.build_grid_utc", fake_build_grid_utc, raising=True
    )

    ctx = DummyCtx()
    start = pd.Timestamp("2020-01-02 00:00:00Z")
    end = pd.Timestamp("2020-01-03 23:59:59Z")
    delay = pd.Timedelta(minutes=5)

    src = SessionCloseEvents(calendar="XNYS", delay=delay)
    out = src.events(ctx, start, end)

    base = pd.date_range(
        start=start.normalize(), end=end.normalize(), freq="D", tz="UTC"
    )
    expected = (base + delay).tz_convert("UTC")
    pd.testing.assert_index_equal(out, expected)


def test_fixed_interval_events_uses_freq_and_delay(monkeypatch):
    calls = []

    def fake_build_grid_utc(cal, start_utc, end_utc, grid):
        calls.append(grid)
        # Make grid reflect the provided freq
        return pd.date_range(start=start_utc, end=end_utc, freq=grid, tz="UTC")

    monkeypatch.setattr(
        "alphaforge.time.events.build_grid_utc", fake_build_grid_utc, raising=True
    )

    ctx = DummyCtx()
    start = pd.Timestamp("2020-01-02 14:30:00Z")
    end = pd.Timestamp("2020-01-02 15:00:00Z")
    delay = pd.Timedelta(seconds=10)

    src = FixedIntervalEvents(calendar="XNYS", freq="5min", delay=delay)
    out = src.events(ctx, start, end)

    # build_grid_utc should be called with the freq string
    assert calls == ["5min"]

    base = pd.date_range(start=start, end=end, freq="5min", tz="UTC")
    expected = (base + delay).tz_convert("UTC")
    pd.testing.assert_index_equal(out, expected)

    # Sanity: tz-aware UTC
    assert out.tz is not None and str(out.tz) in ("UTC", "UTC+00:00")
