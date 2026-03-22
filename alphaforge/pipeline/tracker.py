"""Source health tracker — persists health assessments in PIT."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alphaforge.pit.accessor import PITAccessor, to_utc_aware

from .health import SourceHealthPolicy, SourceHealthStatus, assess_source_health

_STATUS_CODE = {"ok": 0, "late": 1, "stale": 2, "dead": 3, "empty": 4}


@dataclass
class SourceHealthTracker:
    """Track source health over time in the PIT store."""

    pit: PITAccessor
    policies: dict[str, SourceHealthPolicy]

    # Mapping from source_name to series_key prefix used to detect latest obs
    source_prefixes: dict[str, str] | None = None

    def _prefix(self, source_name: str) -> str:
        if self.source_prefixes and source_name in self.source_prefixes:
            return self.source_prefixes[source_name]
        return source_name.replace("_", ".")

    def _latest_obs_date(self, source_name: str) -> pd.Timestamp | None:
        prefix = self._prefix(source_name)
        try:
            result = self.pit.conn.execute(
                "SELECT MAX(obs_date) FROM pit_observations WHERE source = ?",
                [source_name],
            ).fetchone()
        except Exception:
            return None
        if result is None or result[0] is None:
            return None
        ts = pd.Timestamp(result[0])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts

    def assess(self, source_name: str, asof: pd.Timestamp) -> SourceHealthStatus:
        """Assess current health of a source by querying PIT for latest obs."""
        if source_name not in self.policies:
            return SourceHealthStatus(
                source_name=source_name,
                latest_obs_date=None,
                asof=asof,
                age=None,
                age_days=None,
                status="empty",
                weight_factor=0.0,
                expected_next=None,
                message=f"{source_name}: no policy configured",
            )
        latest = self._latest_obs_date(source_name)
        return assess_source_health(source_name, latest, asof, self.policies[source_name])

    def assess_all(self, asof: pd.Timestamp) -> dict[str, SourceHealthStatus]:
        """Assess all registered sources."""
        return {name: self.assess(name, asof) for name in self.policies}

    def record(self, status: SourceHealthStatus) -> None:
        """Persist a health assessment into PIT."""
        base = f"health.{status.source_name}"
        asof = status.asof
        obs_date = asof

        rows = [
            {
                "series_key": f"{base}.status_code",
                "obs_date": obs_date,
                "asof_utc": asof,
                "value": float(_STATUS_CODE.get(status.status, 4)),
                "source": "health",
            },
            {
                "series_key": f"{base}.weight_factor",
                "obs_date": obs_date,
                "asof_utc": asof,
                "value": status.weight_factor,
                "source": "health",
            },
        ]
        if status.age_days is not None:
            rows.append({
                "series_key": f"{base}.age_days",
                "obs_date": obs_date,
                "asof_utc": asof,
                "value": status.age_days,
                "source": "health",
            })

        df = pd.DataFrame(rows)
        self.pit.upsert_pit_observations(df, strict="coerce")

    def history(
        self,
        source_name: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Retrieve health assessment history for a source."""
        prefix = f"health.{source_name}.%"
        params: list = [prefix]
        filters = ["series_key LIKE ?"]

        if start is not None:
            filters.append("obs_date >= ?")
            params.append(start)
        if end is not None:
            filters.append("obs_date <= ?")
            params.append(end)

        where = " AND ".join(filters)
        result = self.pit.conn.execute(
            f"SELECT series_key, obs_date, asof_utc, value FROM pit_observations WHERE {where} ORDER BY obs_date",
            params,
        ).fetchdf()
        return result

    def is_any_degraded(self, asof: pd.Timestamp) -> bool:
        """Quick check: is any source late/stale/dead?"""
        for name in self.policies:
            status = self.assess(name, asof)
            if status.status in ("late", "stale", "dead", "empty"):
                return True
        return False


# Generalized alias
HealthTracker = SourceHealthTracker
