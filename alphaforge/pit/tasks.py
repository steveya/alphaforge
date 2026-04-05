from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import pandas as pd

from alphaforge.time.ref_period import RefPeriod

from .accessor import PITAccessor, to_utc_aware, to_utc_naive
from .exceptions import PITContractError
from .models import PITFoldSpec, PITTapeSpec, coerce_pit_tape_spec


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


def _normalize_asof_grid(
    asof_grid: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(asof_grid, utc=True))
    if len(idx) == 0:
        raise PITContractError("asof_grid must contain at least one timestamp.")
    return idx.sort_values().unique()


def _resolve_snapshot_window(
    start_ref: object | None,
    end_ref: object | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_obs = RefPeriod.parse(start_ref).end_obs_date() if start_ref is not None else None
    end_obs = RefPeriod.parse(end_ref).end_obs_date() if end_ref is not None else None
    return start_obs, end_obs


def iter_walk_forward_folds(
    asof_grid: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    train_size: int | None = None,
    min_train_size: int | None = None,
    validation_size: int = 1,
    step: int = 1,
    purge: int = 0,
) -> Iterator[PITFoldSpec]:
    grid = _normalize_asof_grid(asof_grid)
    if validation_size <= 0:
        raise PITContractError("validation_size must be > 0.")
    if step <= 0:
        raise PITContractError("step must be > 0.")
    if purge < 0:
        raise PITContractError("purge must be >= 0.")
    if train_size is not None and train_size <= 0:
        raise PITContractError("train_size must be > 0 when provided.")
    if min_train_size is not None and min_train_size <= 0:
        raise PITContractError("min_train_size must be > 0 when provided.")
    if train_size is None and min_train_size is None:
        raise PITContractError("Provide train_size or min_train_size for walk-forward folds.")

    required_train = train_size if train_size is not None else min_train_size
    assert required_train is not None

    fold_num = 0
    latest_start = len(grid) - validation_size
    for validation_start in range(required_train + purge, latest_start + 1, step):
        train_stop = validation_start - purge
        if train_stop <= 0:
            continue
        if train_size is None:
            train_slice = grid[:train_stop]
            if min_train_size is not None and len(train_slice) < min_train_size:
                continue
        else:
            train_start = train_stop - train_size
            if train_start < 0:
                continue
            train_slice = grid[train_start:train_stop]
        validation_slice = grid[validation_start : validation_start + validation_size]
        if len(train_slice) == 0 or len(validation_slice) == 0:
            continue
        yield PITFoldSpec(
            fold_id=f"walk_forward_{fold_num:03d}",
            fold_mode="walk_forward",
            train_asofs=tuple(pd.Timestamp(ts) for ts in train_slice),
            validation_asofs=tuple(pd.Timestamp(ts) for ts in validation_slice),
            purge=purge,
            embargo=0,
        )
        fold_num += 1


def iter_purged_kfold_folds(
    asof_grid: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[PITFoldSpec]:
    grid = _normalize_asof_grid(asof_grid)
    if n_splits <= 1:
        raise PITContractError("n_splits must be > 1.")
    if n_splits > len(grid):
        raise PITContractError("n_splits cannot exceed the number of as-of timestamps.")
    if purge < 0:
        raise PITContractError("purge must be >= 0.")
    if embargo < 0:
        raise PITContractError("embargo must be >= 0.")

    fold_sizes = [len(grid) // n_splits] * n_splits
    for idx in range(len(grid) % n_splits):
        fold_sizes[idx] += 1

    start_idx = 0
    for fold_num, fold_size in enumerate(fold_sizes):
        stop_idx = start_idx + fold_size
        validation_slice = grid[start_idx:stop_idx]
        left_stop = max(0, start_idx - purge)
        right_start = min(len(grid), stop_idx + embargo)
        train_slice = grid[:left_stop].append(grid[right_start:])

        yield PITFoldSpec(
            fold_id=f"purged_kfold_{fold_num:03d}",
            fold_mode="purged_kfold",
            train_asofs=tuple(pd.Timestamp(ts) for ts in train_slice),
            validation_asofs=tuple(pd.Timestamp(ts) for ts in validation_slice),
            purge=purge,
            embargo=embargo,
        )
        start_idx = stop_idx


def build_snapshot_tape(
    pit: PITAccessor,
    spec: PITTapeSpec | Mapping[str, object],
    *,
    allow_research: bool = False,
) -> pd.DataFrame:
    tape_spec = coerce_pit_tape_spec(spec)
    if tape_spec.mode == "smoothed_research" and not allow_research:
        raise PITContractError(
            "mode='smoothed_research' requires allow_research=True."
        )

    terminal_asof = (
        max(tape_spec.step_asofs)
        if tape_spec.mode == "smoothed_research" and tape_spec.terminal_asof is None
        else tape_spec.terminal_asof
    )

    rows: list[pd.DataFrame] = []
    for step_asof in tape_spec.step_asofs:
        materialized_asof = (
            pd.Timestamp(step_asof)
            if tape_spec.mode == "filtered"
            else pd.Timestamp(terminal_asof)
        )
        for series_spec in tape_spec.series_specs:
            start_obs, end_obs = _resolve_snapshot_window(
                series_spec.start_ref,
                series_spec.end_ref,
            )
            snapshot_rows = pit._snapshot_rows_with_release_policy(
                series_spec.series_key,
                materialized_asof,
                policy=series_spec.release_policy,
                start=start_obs,
                end=end_obs,
            )
            if snapshot_rows.empty:
                continue

            frame = snapshot_rows.copy()
            frame["step_asof_utc"] = pd.Timestamp(step_asof)
            frame["materialized_asof_utc"] = materialized_asof
            frame["series_key"] = series_spec.series_key
            frame["series_alias"] = series_spec.alias or series_spec.series_key
            frame["sequence_mode"] = tape_spec.mode
            rows.append(
                frame[
                    [
                        "step_asof_utc",
                        "materialized_asof_utc",
                        "obs_date",
                        "series_key",
                        "series_alias",
                        "value",
                        "source_asof_utc",
                        "sequence_mode",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "step_asof_utc",
                "materialized_asof_utc",
                "obs_date",
                "series_key",
                "series_alias",
                "value",
                "source_asof_utc",
                "sequence_mode",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    for column in ["step_asof_utc", "materialized_asof_utc", "obs_date", "source_asof_utc"]:
        out[column] = pd.to_datetime(out[column], utc=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.sort_values(
        ["step_asof_utc", "series_alias", "obs_date", "source_asof_utc"]
    ).reset_index(drop=True)
