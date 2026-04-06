from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .registry_api import RegistryApiSourceBase
from .schema_helpers import single_value_schema


class EIADataSource(RegistryApiSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._api_key = api_key or os.getenv("EIA_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._init_registry(
            "eia_series.yaml",
            registry_entries=registry_entries,
            registry_path=registry_path,
        )

    def schemas(self):
        return {
            self.TABLE: single_value_schema(self.TABLE)
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
        self._require_table(q)
        if not self._api_key:
            raise ValueError("EIA API key required via EIA_API_KEY or constructor arg")
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for entity, cfg in self._iter_entity_configs(
            q,
            error_message="EIADataSource requires q.entities registry keys",
        ):
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
                        "asof_utc": asof_utc,
                    }
                )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
