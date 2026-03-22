"""Thin adapter wrapper around AlphaForge PIT APIs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.pit.transforms import PITTransformResult, PITTransformSpec
from alphaforge.pit.utils.timestamps import coerce_utc_timestamp, normalize_utc_day
from alphaforge.time.ref_period import RefFreq, RefPeriod


class AlphaForgePITLayer:
    """Wrapper for AlphaForge PIT accessors."""

    def __init__(self, ctx: DataContext) -> None:
        if ctx.pit is None:
            raise ValueError("PIT requires DuckDBParquetStore-backed DataContext")
        self._ctx = ctx

    def snapshot(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.Series:
        return self._ctx.pit.get_snapshot(series_key, asof=asof, start=start, end=end)

    def snapshot_multi(
        self,
        series_keys: Iterable[str],
        *,
        asof: pd.Timestamp,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if self._ctx.pit is None:
            raise ValueError("PIT store is not available; cannot query snapshots.")
        return self._ctx.pit.get_snapshot_multi(
            list(series_keys),
            asof=asof,
            start=start,
            end=end,
        )

    def snapshot_ref(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: RefFreq | None = None,
    ) -> pd.Series:
        if isinstance(start_ref, str):
            start_ref = RefPeriod.parse(start_ref)
        if isinstance(end_ref, str):
            end_ref = RefPeriod.parse(end_ref)
        return self._ctx.pit.get_snapshot_ref(
            series_key,
            asof=asof,
            start_ref=start_ref,
            end_ref=end_ref,
            freq=freq,
        )

    def revisions(
        self,
        series_key: str,
        obs_date: pd.Timestamp,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.Series:
        return self._ctx.pit.get_revision_timeline(
            series_key, obs_date=obs_date, start_asof=start_asof, end_asof=end_asof
        )

    def revision_path(
        self,
        series_key: str,
        obs_date: pd.Timestamp,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self._ctx.pit.get_revision_path(
            series_key, obs_date=obs_date, start_asof=start_asof, end_asof=end_asof
        )

    def revision_path_multi(self, requests: pd.DataFrame) -> pd.DataFrame:
        return self._ctx.pit.get_revision_path_multi(requests)

    def revisions_ref(
        self,
        series_key: str,
        ref: str | RefPeriod,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        freq: RefFreq | None = None,
    ) -> pd.Series:
        if isinstance(ref, str):
            ref = RefPeriod.parse(ref)
        return self._ctx.pit.get_revision_timeline_ref(
            series_key, ref=ref, start_asof=start_asof, end_asof=end_asof, freq=freq
        )

    def upsert(self, df: pd.DataFrame) -> None:
        self._ctx.pit.upsert_pit_observations(df)

    def apply_transform(
        self,
        spec: PITTransformSpec | dict[str, Any],
        *,
        overwrite: bool = False,
        persist: bool = True,
        allow_experimental: bool = False,
        on_engine_mismatch: str = "error",
    ) -> PITTransformResult:
        return self._ctx.pit.apply_transform(
            spec,
            overwrite=overwrite,
            persist=persist,
            allow_experimental=allow_experimental,
            on_engine_mismatch=on_engine_mismatch,
        )

    def explain_transform(
        self,
        spec: PITTransformSpec | dict[str, Any],
        *,
        allow_experimental: bool = False,
        on_engine_mismatch: str = "error",
    ) -> dict[str, Any]:
        return self._ctx.pit.explain_transform(
            spec,
            allow_experimental=allow_experimental,
            on_engine_mismatch=on_engine_mismatch,
        )

    def list_pit_observations_asof(
        self,
        *,
        series_key: str,
        obs_date: date,
        asof_date: date,
    ) -> pd.DataFrame:
        if self._ctx.pit is None:
            raise ValueError("PIT store is not available; cannot list PIT observations.")
        conn = self._ctx.pit.conn
        obs_day = normalize_utc_day(obs_date)
        asof_cutoff = coerce_utc_timestamp(asof_date).normalize() + pd.Timedelta(days=1)
        df = conn.execute(
            """
            SELECT series_key, obs_date, asof_utc, value
            FROM pit_observations
            WHERE series_key = ?
              AND DATE(obs_date) = ?
              AND asof_utc < ?
            ORDER BY asof_utc ASC
            """,
            [series_key, obs_day, asof_cutoff],
        ).fetchdf()
        if df.empty:
            return pd.DataFrame(
                {
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        df["obs_date"] = pd.to_datetime(df["obs_date"], utc=True).dt.floor("D")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        return df

    def list_pit_observations_asof_multi(self, requests: pd.DataFrame) -> pd.DataFrame:
        """Batch list PIT observations for multiple (series_key, obs_date, asof_date) requests."""
        required = {"request_id", "series_key", "obs_date", "asof_date"}
        missing = required - set(requests.columns)
        if missing:
            raise ValueError(f"Missing required request columns: {sorted(missing)}")

        if self._ctx.pit is None:
            raise ValueError("PIT store is not available; cannot list PIT observations.")

        _empty = pd.DataFrame(
            {
                "request_id": pd.Series(dtype="object"),
                "series_key": pd.Series(dtype="object"),
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )

        if requests.empty:
            return _empty

        req = requests.loc[:, ["request_id", "series_key", "obs_date", "asof_date"]].copy()
        req["obs_day"] = pd.to_datetime(req["obs_date"], utc=True, errors="coerce").dt.date
        req["asof_cutoff"] = (
            pd.to_datetime(req["asof_date"], utc=True, errors="coerce").dt.normalize()
            + pd.Timedelta(days=1)
        ).dt.tz_localize(None)
        req = req.dropna(subset=["obs_day", "asof_cutoff", "series_key", "request_id"])
        if req.empty:
            return _empty

        conn = self._ctx.pit.conn
        conn.register("pit_asof_requests", req)
        try:
            df = conn.execute(
                """
                SELECT
                    r.request_id,
                    p.series_key,
                    p.obs_date,
                    p.asof_utc,
                    p.value
                FROM pit_asof_requests r
                LEFT JOIN pit_observations p
                    ON p.series_key = r.series_key
                   AND DATE(p.obs_date) = r.obs_day
                   AND p.asof_utc < r.asof_cutoff
                ORDER BY r.request_id, p.asof_utc
                """
            ).fetchdf()
        finally:
            conn.unregister("pit_asof_requests")

        if df.empty:
            return _empty

        df = df[df["series_key"].notna()].copy()
        if df.empty:
            return _empty
        df["obs_date"] = pd.to_datetime(df["obs_date"], utc=True).dt.floor("D")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.reset_index(drop=True)
