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


class DestatisGenesisDataSource(DataSource):
    name = "destatis_genesis"
    TABLE = "destatis_series"

    def __init__(
        self,
        *,
        user: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://www-genesis.destatis.de/genesisWS/rest/2020",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self._user = user or os.getenv("DESTATIS_GENESIS_USER")
        self._password = password or os.getenv("DESTATIS_GENESIS_PASS")
        self._api_key = api_key or os.getenv("DESTATIS_GENESIS_KEY")
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        entries = load_registry_entries(
            "destatis_series.yaml",
            entries=registry_entries,
            registry_path=registry_path,
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
                native_freq="M",
            )
        }

    @staticmethod
    def _parse_period(period: str) -> pd.Timestamp | None:
        txt = str(period)
        if len(txt) == 6 and txt.isdigit():
            return pd.Timestamp(
                year=int(txt[:4]), month=int(txt[4:6]), day=1, tz="UTC"
            ) + pd.offsets.MonthEnd(0)
        return pd.to_datetime(txt, errors="coerce", utc=True)

    def _call(self, cfg: dict) -> dict:
        params = dict(cfg.get("params", {}))
        params.setdefault("name", "data")
        params.setdefault("format", "json")
        if self._api_key:
            params["apikey"] = self._api_key
        if self._user:
            params["username"] = self._user
        if self._password:
            params["password"] = self._password
        params.setdefault("table", cfg.get("table_code", ""))
        url = f"{self._base_url}/{params.pop('name')}?{urlencode(params, doseq=True)}"
        payload = self._http.get_bytes(
            url=url,
            source="destatis",
            artifact_name=f"{cfg.get('table_code','table')}.json",
        )
        return json.loads(payload.decode("utf-8", errors="ignore"))

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError(
                "DestatisGenesisDataSource requires q.entities registry keys"
            )

        rows = []
        for entity in entities:
            cfg = self._registry.get(str(entity))
            if cfg is None:
                continue
            payload = self._call(cfg)
            values = (
                payload.get("Object", {}).get("Value", [])
                if isinstance(payload, dict)
                else []
            )
            if not values and isinstance(payload, dict):
                values = payload.get("value", [])
            for row in values:
                period = row.get("time") or row.get("Zeit") or row.get("period")
                date = self._parse_period(str(period))
                if date is None or pd.isna(date):
                    continue
                value = row.get("value") or row.get("Wert")
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
