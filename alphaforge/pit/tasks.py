from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .accessor import PITAccessor, to_utc_aware, to_utc_naive
from .exceptions import PITContractError


@dataclass(frozen=True)
class RevisionStability:
    obs_date: pd.Timestamp
    n_vintages: int
    total_abs_revision: float
    revision_std: float


@dataclass(frozen=True)
class RevisionEvent:
    obs_date: pd.Timestamp
    asof_utc: pd.Timestamp
    value: float
    delta: float


def first_vintage_snapshot(
    pit: PITAccessor,
    series_key: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    filters = ["series_key = ?"]
    params: list[object] = [series_key]
    if start is not None:
        filters.append("obs_date >= ?")
        params.append(to_utc_naive(start))
    if end is not None:
        filters.append("obs_date <= ?")
        params.append(to_utc_naive(end))

    where_clause = " AND ".join(filters)
    query = f"""
        SELECT obs_date, value
        FROM (
            SELECT
                obs_date,
                value,
                ROW_NUMBER() OVER (
                    PARTITION BY obs_date
                    ORDER BY asof_utc ASC
                ) AS rn
            FROM pit_observations
            WHERE {where_clause}
        ) ranked
        WHERE rn = 1
        ORDER BY obs_date
    """
    df = pit.conn.execute(query, params).fetchdf()
    if df.empty:
        return pd.Series(dtype="float64", name=series_key)
    out = pd.Series(df["value"].to_numpy(), index=to_utc_aware(df["obs_date"]), name=series_key)
    out.index.name = "obs_date"
    return out


def latest_vintage_snapshot(
    pit: PITAccessor,
    series_key: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    filters = ["series_key = ?"]
    params: list[object] = [series_key]
    if start is not None:
        filters.append("obs_date >= ?")
        params.append(to_utc_naive(start))
    if end is not None:
        filters.append("obs_date <= ?")
        params.append(to_utc_naive(end))

    where_clause = " AND ".join(filters)
    query = f"""
        SELECT obs_date, value
        FROM (
            SELECT
                obs_date,
                value,
                ROW_NUMBER() OVER (
                    PARTITION BY obs_date
                    ORDER BY asof_utc DESC
                ) AS rn
            FROM pit_observations
            WHERE {where_clause}
        ) ranked
        WHERE rn = 1
        ORDER BY obs_date
    """
    df = pit.conn.execute(query, params).fetchdf()
    if df.empty:
        return pd.Series(dtype="float64", name=series_key)
    out = pd.Series(df["value"].to_numpy(), index=to_utc_aware(df["obs_date"]), name=series_key)
    out.index.name = "obs_date"
    return out


def snapshot_at_horizon(
    pit: PITAccessor,
    series_key: str,
    horizon: pd.Timedelta,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    base = latest_vintage_snapshot(pit, series_key, start=start, end=end)
    if base.empty:
        return base

    values: list[float] = []
    idx: list[pd.Timestamp] = []

    for obs_date in base.index:
        cutoff = obs_date + horizon
        snap = pit.get_snapshot(series_key, cutoff, start=obs_date, end=obs_date)
        if snap.empty:
            continue
        idx.append(obs_date)
        values.append(float(snap.iloc[0]))

    if not values:
        return pd.Series(dtype="float64", name=series_key)
    out = pd.Series(values, index=pd.DatetimeIndex(idx), name=series_key)
    out.index.name = "obs_date"
    return out


def revision_deltas(
    pit: PITAccessor,
    series_key: str,
    obs_date: pd.Timestamp,
) -> pd.Series:
    timeline = pit.get_revision_timeline(series_key, obs_date)
    if timeline.empty:
        return pd.Series(dtype="float64", name=f"{series_key}_delta")
    out = timeline.diff().rename(f"{series_key}_delta")
    return out


def revision_events(
    pit: PITAccessor,
    series_key: str,
    obs_date: pd.Timestamp,
    *,
    min_abs_change: float = 0.0,
) -> pd.DataFrame:
    timeline = pit.get_revision_timeline(series_key, obs_date)
    if timeline.empty:
        return pd.DataFrame(columns=["asof_utc", "value", "delta"])

    df = pd.DataFrame({"asof_utc": timeline.index, "value": timeline.values})
    df["delta"] = df["value"].diff()
    if min_abs_change > 0:
        df = df[df["delta"].abs() >= float(min_abs_change)]
    return df.reset_index(drop=True)


def revision_stability(
    pit: PITAccessor,
    series_key: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    latest = latest_vintage_snapshot(pit, series_key, start=start, end=end)
    if latest.empty:
        return pd.DataFrame(
            columns=["obs_date", "n_vintages", "total_abs_revision", "revision_std"]
        )

    rows: list[RevisionStability] = []
    for obs_date in latest.index:
        tl = pit.get_revision_timeline(series_key, obs_date)
        if tl.empty:
            continue
        deltas = tl.diff().dropna()
        rows.append(
            RevisionStability(
                obs_date=obs_date,
                n_vintages=int(len(tl)),
                total_abs_revision=float(deltas.abs().sum()) if not deltas.empty else 0.0,
                revision_std=float(deltas.std()) if len(deltas) > 1 else 0.0,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["obs_date", "n_vintages", "total_abs_revision", "revision_std"]
        )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("obs_date").reset_index(drop=True)
    return df


def _obs_dates_for_series(
    pit: PITAccessor,
    series_key: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    filters = ["series_key = ?"]
    params: list[object] = [series_key]
    if start is not None:
        filters.append("obs_date >= ?")
        params.append(to_utc_naive(start))
    if end is not None:
        filters.append("obs_date <= ?")
        params.append(to_utc_naive(end))

    where_clause = " AND ".join(filters)
    df = pit.conn.execute(
        f"""
        SELECT DISTINCT obs_date
        FROM pit_observations
        WHERE {where_clause}
        ORDER BY obs_date
        """,
        params,
    ).fetchdf()
    if df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(to_utc_aware(df["obs_date"]))


def revision_event_stream(
    pit: PITAccessor,
    series_key: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    min_abs_change: float = 0.0,
) -> pd.DataFrame:
    """Return revision events across all obs_date timelines."""
    if min_abs_change < 0:
        raise PITContractError("min_abs_change must be >= 0.")

    events: list[RevisionEvent] = []
    for obs_date in _obs_dates_for_series(pit, series_key, start=start, end=end):
        ev = revision_events(
            pit,
            series_key,
            obs_date,
            min_abs_change=min_abs_change,
        )
        if ev.empty:
            continue
        for row in ev.itertuples(index=False):
            if pd.isna(row.delta):
                continue
            events.append(
                RevisionEvent(
                    obs_date=obs_date,
                    asof_utc=pd.Timestamp(row.asof_utc),
                    value=float(row.value),
                    delta=float(row.delta),
                )
            )

    if not events:
        return pd.DataFrame(columns=["obs_date", "asof_utc", "value", "delta"])
    out = pd.DataFrame([e.__dict__ for e in events])
    return out.sort_values(["obs_date", "asof_utc"]).reset_index(drop=True)


def revision_volatility(
    pit: PITAccessor,
    series_key: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Compute standard deviation of revision deltas by obs_date."""
    stability = revision_stability(pit, series_key, start=start, end=end)
    if stability.empty:
        return pd.Series(dtype="float64", name=f"{series_key}_revision_volatility")
    out = pd.Series(
        stability["revision_std"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(stability["obs_date"], utc=True)),
        name=f"{series_key}_revision_volatility",
    )
    out.index.name = "obs_date"
    return out


def forward_fill_with_staleness(
    snapshot: pd.Series,
    *,
    max_staleness: pd.Timedelta,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    s = snapshot.sort_index()
    if target_index is None:
        target_index = pd.DatetimeIndex(s.index)

    target_index = pd.DatetimeIndex(pd.to_datetime(target_index))
    if target_index.tz is None:
        target_index = target_index.tz_localize("UTC")
    else:
        target_index = target_index.tz_convert("UTC")

    expanded = s.reindex(target_index).ffill()

    source_dates = pd.Series(s.index, index=s.index)
    last_source = source_dates.reindex(target_index).ffill()
    age = target_index - pd.DatetimeIndex(last_source)

    is_stale = age > max_staleness
    values = expanded.copy()
    values[is_stale] = pd.NA

    return pd.DataFrame(
        {
            "value": values,
            "source_obs_date": pd.DatetimeIndex(last_source),
            "age": age,
            "is_stale": is_stale,
            "age_days": age / pd.Timedelta(days=1),
        },
        index=target_index,
    )


def yoy(snapshot: pd.Series, periods: int = 12) -> pd.Series:
    return snapshot.sort_index().pct_change(periods=periods)


def qoq(snapshot: pd.Series, periods: int = 1) -> pd.Series:
    return snapshot.sort_index().pct_change(periods=periods)
