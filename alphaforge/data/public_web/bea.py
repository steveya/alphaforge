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


class BEADataSource(RegistryApiSourceBase):
    name = "bea"
    TABLE = "bea_series"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        api_url: str = "https://apps.bea.gov/api/data",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._api_key = api_key or os.getenv("BEA_API_KEY")
        self._api_url = api_url
        self._init_registry(
            "bea_series.yaml",
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
        if "Q" in txt:
            year_txt, quarter_txt = txt.split("Q")
            month_num = int(quarter_txt) * 3
            return pd.Timestamp(
                year=int(year_txt), month=month_num, day=1, tz="UTC"
            ) + pd.offsets.MonthEnd(0)
        if "-" in txt:
            year_txt, month_txt = txt.split("-")[:2]
            return pd.Timestamp(
                year=int(year_txt), month=int(month_txt), day=1, tz="UTC"
            ) + pd.offsets.MonthEnd(0)
        if len(txt) == 4 and txt.isdigit():
            return pd.Timestamp(year=int(txt), month=12, day=31, tz="UTC")
        return None

    def _call(self, params: dict) -> dict:
        query = {"UserID": self._api_key, "method": "GetData", **params}
        url = f"{self._api_url}?{urlencode(query, doseq=True)}"
        payload = self._http.get_bytes(
            url=url, source="bea", artifact_name=f"{params.get('TableName','bea')}.json"
        )
        return json.loads(payload.decode("utf-8"))

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)
        if not self._api_key:
            raise ValueError("BEA API key required via BEA_API_KEY or constructor arg")
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for entity, config in self._iter_entity_configs(
            q,
            error_message="BEADataSource requires q.entities registry keys",
        ):
            params = dict(config.get("params", {}))
            payload = self._call(params)
            data_rows = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
            for row in data_rows:
                date = self._parse_period(str(row.get("TimePeriod", "")))
                if date is None:
                    continue
                rows.append(
                    {
                        "date": date,
                        "entity_id": str(entity),
                        "value": pd.to_numeric(
                            str(row.get("DataValue", "")).replace(",", ""),
                            errors="coerce",
                        ),
                        "asof_utc": asof_utc,
                    }
                )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
