from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource
from alphaforge.pit.accessor import PITAccessor, to_utc_aware, to_utc_naive
from alphaforge.pit.guards import ReleaseLagPolicy, effective_asof


@dataclass
class PITDataSource(DataSource):
    """Expose PIT snapshots/observations through the DataSource contract."""

    pit: PITAccessor
    lag_policy: ReleaseLagPolicy | None = None
    name: str = "pit"

    SNAPSHOT_TABLE = "pit.snapshot"
    OBS_TABLE = "pit.observations"

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.SNAPSHOT_TABLE: TableSchema(
                name=self.SNAPSHOT_TABLE,
                required_columns=["value", "asof_utc"],
                canonical_columns=["value", "asof_utc", "source", "meta_json"],
                entity_column="entity_id",
                time_column="date",
                native_freq=None,
                time_semantics="interval_end",
                release_time_column="asof_utc",
            ),
            self.OBS_TABLE: TableSchema(
                name=self.OBS_TABLE,
                required_columns=["value", "asof_utc"],
                canonical_columns=[
                    "value",
                    "asof_utc",
                    "release_time_utc",
                    "revision_id",
                    "source",
                    "meta_json",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq=None,
                time_semantics="interval_end",
                release_time_column="asof_utc",
                revision_id_column="revision_id",
            ),
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table == self.SNAPSHOT_TABLE:
            return self._fetch_snapshot(q)
        if q.table == self.OBS_TABLE:
            return self._fetch_observations(q)
        raise KeyError(f"Unsupported PIT table: {q.table}")

    def _list_series_keys(self, entities: list[str] | None = None) -> list[str]:
        if entities is not None:
            return [str(x) for x in entities]
        df = self.pit.conn.execute(
            "SELECT DISTINCT series_key FROM pit_observations ORDER BY series_key"
        ).fetchdf()
        return [str(x) for x in df["series_key"].tolist()]

    @staticmethod
    def _project(df: pd.DataFrame, q: Query) -> pd.DataFrame:
        required = ["date", "entity_id"]
        cols = required + [c for c in q.columns if c not in required]
        out = df.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[cols]

    def _fetch_snapshot(self, q: Query) -> pd.DataFrame:
        if q.asof is None:
            raise ValueError("pit.snapshot requires Query.asof.")

        keys = self._list_series_keys(
            [str(x) for x in q.entities] if q.entities is not None else None
        )
        rows: list[dict] = []

        for series_key in keys:
            asof = q.asof
            if self.lag_policy is not None:
                asof = effective_asof(asof, series_key, self.lag_policy)

            snap = self.pit.get_snapshot(series_key, asof, start=q.start, end=q.end)
            if snap.empty:
                continue

            for obs_date, value in snap.items():
                rows.append(
                    {
                        "date": obs_date,
                        "entity_id": series_key,
                        "value": float(value) if pd.notna(value) else pd.NA,
                        "asof_utc": asof,
                        "source": "pit.snapshot",
                        "meta_json": pd.NA,
                    }
                )

        if not rows:
            base = pd.DataFrame(columns=["date", "entity_id"] + list(q.columns))
            return self._project(base, q)

        out = pd.DataFrame(rows)
        out["date"] = to_utc_aware(out["date"])
        out["asof_utc"] = to_utc_aware(out["asof_utc"])
        return self._project(out, q)

    def _fetch_observations(self, q: Query) -> pd.DataFrame:
        filters = ["1=1"]
        params: list[object] = []

        if q.entities is not None and len(q.entities) > 0:
            placeholders = ",".join(["?"] * len(q.entities))
            filters.append(f"series_key IN ({placeholders})")
            params.extend([str(x) for x in q.entities])

        if q.start is not None:
            filters.append("obs_date >= ?")
            params.append(to_utc_naive(q.start))

        if q.end is not None:
            filters.append("obs_date <= ?")
            params.append(to_utc_naive(q.end))

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT
                obs_date AS date,
                series_key AS entity_id,
                asof_utc,
                value,
                release_time_utc,
                revision_id,
                source,
                meta_json
            FROM pit_observations
            WHERE {where_clause}
            ORDER BY series_key, obs_date, asof_utc
        """
        out = self.pit.conn.execute(query, params).fetchdf()
        if out.empty:
            base = pd.DataFrame(columns=["date", "entity_id"] + list(q.columns))
            return self._project(base, q)

        out["date"] = to_utc_aware(out["date"])
        out["asof_utc"] = to_utc_aware(out["asof_utc"])
        if "release_time_utc" in out.columns:
            out["release_time_utc"] = to_utc_aware(out["release_time_utc"])

        if q.asof is not None:
            if self.lag_policy is None:
                cutoff = q.asof
                out = out[out["asof_utc"] <= cutoff]
            else:
                # Per-series effective cutoff
                cutoffs = out["entity_id"].map(
                    lambda s: effective_asof(q.asof, str(s), self.lag_policy)
                )
                out = out[out["asof_utc"] <= cutoffs.to_numpy()]

        return self._project(out, q)
