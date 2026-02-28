from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .registry_loader import load_registry_entries, map_registry
from .utils import apply_query_filters, project_columns


class EIADataSource(DataSource):
    name = "eia"
    TABLE = "eia_series"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://api.eia.gov/v2",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("EIA_API_KEY")
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        entries = load_registry_entries(
            "eia_series.yaml", entries=registry_entries, registry_path=registry_path
        )
        self._registry = map_registry(entries)

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value"],
                entity_column="entity_id",
                time_column="date",
            )
        }

    def _call(self, config: dict, q: Query) -> dict:
        route = str(config.get("route", ""))
        params = dict(config.get("params", {}))
        params["api_key"] = self._api_key
        if q.start is not None:
            params["start"] = q.start.strftime("%Y-%m-%d")
        if q.end is not None:
            params["end"] = q.end.strftime("%Y-%m-%d")
        url = f"{self._base_url}/{route}/data/?{urlencode(params, doseq=True)}"
        payload = self._http.get_bytes(
            url=url, source="eia", artifact_name=f"{route.replace('/','_')}.json"
        )
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _parse_date(value: str) -> pd.Timestamp | None:
        txt = str(value)
        for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
            try:
                ts = pd.Timestamp.strptime(txt, fmt).tz_localize("UTC")
                if fmt == "%Y-%m":
                    ts = ts + pd.offsets.MonthEnd(0)
                elif fmt == "%Y":
                    ts = pd.Timestamp(year=ts.year, month=12, day=31, tz="UTC")
                return ts
            except Exception:
                pass
        return pd.to_datetime(txt, errors="coerce", utc=True)

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        if not self._api_key:
            raise ValueError("EIA API key required via EIA_API_KEY or constructor arg")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError("EIADataSource requires q.entities registry keys")

        rows = []
        for entity in entities:
            cfg = self._registry.get(str(entity))
            if cfg is None:
                continue
            payload = self._call(cfg, q)
            for row in payload.get("response", {}).get("data", []):
                date = self._parse_date(str(row.get("period", "")))
                if date is None or pd.isna(date):
                    continue
                value = row.get("value")
                if value is None:
                    candidates = [v for k, v in row.items() if k not in {"period"}]
                    value = candidates[0] if candidates else None
                rows.append(
                    {
                        "date": date,
                        "entity_id": str(entity),
                        "value": pd.to_numeric(value, errors="coerce"),
                        "asof_utc": q.asof or pd.Timestamp.now(tz="UTC"),
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["date", "entity_id", "asof_utc", "value"])
        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        schema = self.schemas()[self.TABLE]
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col="date",
            entity_col="entity_id",
        )
        return out.sort_values(["entity_id", "date"]).reset_index(drop=True)
