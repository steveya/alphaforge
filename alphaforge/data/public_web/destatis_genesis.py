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


class DestatisGenesisDataSource(RegistryApiSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._user = user or os.getenv("DESTATIS_GENESIS_USER")
        self._password = password or os.getenv("DESTATIS_GENESIS_PASS")
        self._api_key = api_key or os.getenv("DESTATIS_GENESIS_KEY")
        self._base_url = base_url.rstrip("/")
        self._init_registry(
            "destatis_series.yaml",
            registry_entries=registry_entries,
            registry_path=registry_path,
        )

    def schemas(self):
        return {
            self.TABLE: single_value_schema(self.TABLE, native_freq="M")
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
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for entity, cfg in self._iter_entity_configs(
            q,
            error_message="DestatisGenesisDataSource requires q.entities registry keys",
        ):
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
                        "asof_utc": asof_utc,
                    }
                )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
