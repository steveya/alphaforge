"""Adapter for fetching data from AlphaForge PIT storage."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.adapters.alphaforge_layer import AlphaForgePITLayer
from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.observation import PITObservation, SeriesMetadata
from alphaforge.time.ref_period import RefFreq, RefPeriod


def _normalize_obs_date_key(value: object) -> date:
    """Normalize AlphaForge obs_date timestamps to canonical date keys.

    AlphaForge may return obs_date timestamps with non-midnight UTC times.
    We round by adding 12 hours before taking the date to recover period-end keys.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("obs_date is missing or NaT")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts != ts.normalize():
        ts = ts + pd.Timedelta(hours=12)
    return ts.normalize().date()


class AlphaForgePITAdapter(PITAdapter):
    """Point-in-time data adapter for AlphaForge."""

    def __init__(self, ctx: DataContext) -> None:
        if ctx.pit is None:
            raise ValueError("AlphaForge PIT adapter requires PIT-enabled DataContext")
        self._ctx = ctx
        self._pit: PITAccessor = ctx.pit
        self._layer = AlphaForgePITLayer(ctx)

    @property
    def name(self) -> str:
        return "alphaforge"

    def supports_pit(self, series_id: str) -> bool:
        return True

    def list_vintages(self, query_series_key: str) -> list[date]:
        conn = self._pit.conn
        rows = conn.execute(
            "SELECT DISTINCT asof_utc FROM pit_observations WHERE series_key = ?",
            [query_series_key],
        ).fetchall()
        if not rows:
            return []
        return sorted(
            {pd.Timestamp(row[0], tz="UTC").date() for row in rows if row[0] is not None}
        )

    def list_pit_observations_asof(
        self,
        *,
        series_key: str,
        obs_date: date,
        asof_date: date,
    ) -> pd.DataFrame:
        return self._layer.list_pit_observations_asof(
            series_key=series_key,
            obs_date=obs_date,
            asof_date=asof_date,
        )

    def list_pit_observations_asof_multi(self, requests: pd.DataFrame) -> pd.DataFrame:
        return self._layer.list_pit_observations_asof_multi(requests)

    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: date | None = None,
        end: date | None = None,
        *,
        metadata: SeriesMetadata | None = None,
        ingest_from_ctx_source: bool = True,
        **kwargs: Any,
    ) -> list[PITObservation]:
        source_series_id = metadata.source_series_id if metadata else series_id
        query_series_key = metadata.series_key if metadata else series_id

        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

        if ingest_from_ctx_source and "fred" in self._ctx.sources:
            query = Query(
                table="fred_series",
                columns=["value"],
                entities=[source_series_id],
                start=start_ts,
                end=end_ts,
                asof=asof_ts,
            )
            panel = self._ctx.fetch_panel("fred", query)
            panel_df = panel.df.reset_index()
            required = {"entity_id", "ts_utc", "asof_utc", "value"}
            missing = required - set(panel_df.columns)
            if missing:
                raise ValueError(
                    f"Unexpected alphaforge panel schema; missing {sorted(missing)}"
                )

            if metadata is None:
                series_key_values = panel_df["entity_id"]
            else:
                series_key_values = [query_series_key] * len(panel_df)
            pit_df = pd.DataFrame(
                {
                    "series_key": series_key_values,
                    "obs_date": pd.to_datetime(panel_df["ts_utc"], utc=True).dt.floor("D"),
                    "asof_utc": pd.to_datetime(panel_df["asof_utc"], utc=True),
                    "value": panel_df["value"],
                    "source": pd.NA,
                    "revision_id": pd.NA,
                    "meta_json": pd.NA,
                    "release_time_utc": pd.NaT,
                }
            )
            self._pit.upsert_pit_observations(pit_df)

        snap = self._layer.snapshot(query_series_key, asof=asof_ts, start=start_ts, end=end_ts)

        observations = []
        series_key = metadata.series_key if metadata else series_id
        source_series_id = metadata.source_series_id if metadata else series_id
        frequency = metadata.frequency if metadata else ""
        source = metadata.source if metadata else "alphaforge"
        for obs_date_val, value in snap.items():
            obs = PITObservation(
                series_key=series_key,
                source=source,
                source_series_id=source_series_id,
                asof_date=asof_date,
                vintage_date=asof_date,
                obs_date=_normalize_obs_date_key(obs_date_val),
                value=float(value),
                frequency=frequency,
            )
            observations.append(obs)
        return observations

    def fetch_asof_many(
        self,
        *,
        series_keys: Iterable[str],
        asof_date: date,
        start: date | None = None,
        end: date | None = None,
        metadata_by_key: dict[str, SeriesMetadata] | None = None,
        ingest_from_ctx_source: bool = True,
    ) -> dict[str, list[PITObservation]]:
        """Fetch multiple series snapshots at once using a single PIT query."""
        keys = [str(key) for key in series_keys]
        if not keys:
            return {}

        if ingest_from_ctx_source:
            return {
                key: self.fetch_asof(
                    key,
                    asof_date,
                    start=start,
                    end=end,
                    metadata=(metadata_by_key or {}).get(key),
                    ingest_from_ctx_source=ingest_from_ctx_source,
                )
                for key in keys
            }

        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None
        batch = self._layer.snapshot_multi(keys, asof=asof_ts, start=start_ts, end=end_ts)
        grouped: dict[str, list[PITObservation]] = {key: [] for key in keys}

        if batch.empty:
            return grouped

        metadata_map = metadata_by_key or {}
        for row in batch.itertuples(index=False):
            query_series_key = str(row.series_key)
            value = row.value
            obs_date_val = _normalize_obs_date_key(row.obs_date)
            if pd.isna(value):
                continue

            meta = metadata_map.get(query_series_key)
            source_series_id = meta.source_series_id if meta else query_series_key
            frequency = meta.frequency if meta else ""
            source = meta.source if meta else "alphaforge"
            obs = PITObservation(
                series_key=query_series_key,
                source=source,
                source_series_id=source_series_id,
                asof_date=asof_date,
                vintage_date=asof_date,
                obs_date=obs_date_val,
                value=float(value),
                frequency=frequency,
            )
            grouped.setdefault(query_series_key, []).append(obs)

        return grouped

    def fetch_asof_ref(
        self,
        series_id: str,
        asof_date: date,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: RefFreq | None = None,
        metadata: SeriesMetadata | None = None,
    ) -> list[PITObservation]:
        query_series_key = metadata.series_key if metadata else series_id
        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        snap = self._layer.snapshot_ref(
            query_series_key, asof=asof_ts, start_ref=start_ref, end_ref=end_ref, freq=freq
        )
        observations = []
        series_key = metadata.series_key if metadata else series_id
        source_series_id = metadata.source_series_id if metadata else series_id
        source = metadata.source if metadata else "alphaforge"
        frequency = metadata.frequency if metadata else (freq.value if freq else "")
        for obs_date_val, value in snap.items():
            obs = PITObservation(
                series_key=series_key,
                source=source,
                source_series_id=source_series_id,
                asof_date=asof_date,
                vintage_date=asof_date,
                obs_date=_normalize_obs_date_key(obs_date_val),
                value=float(value),
                frequency=frequency,
            )
            observations.append(obs)
        return observations

    def fetch_revisions_ref(
        self,
        series_id: str,
        ref: str | RefPeriod,
        start_asof: date | None = None,
        end_asof: date | None = None,
        *,
        freq: RefFreq | None = None,
        metadata: SeriesMetadata | None = None,
    ) -> pd.Series:
        start_ts = pd.Timestamp(start_asof, tz="UTC") if start_asof else None
        end_ts = pd.Timestamp(end_asof, tz="UTC") if end_asof else None
        query_series_key = metadata.series_key if metadata else series_id
        series = self._layer.revisions_ref(
            query_series_key, ref, start_asof=start_ts, end_asof=end_ts, freq=freq
        )
        if metadata is not None:
            series = series.rename(metadata.series_key)
        return series
