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


class BEADataSource(DataSource):
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
        self._api_key = api_key or os.getenv("BEA_API_KEY")
        self._api_url = api_url
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        entries = load_registry_entries(
            "bea_series.yaml", entries=registry_entries, registry_path=registry_path
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
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        if not self._api_key:
            raise ValueError("BEA API key required via BEA_API_KEY or constructor arg")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError("BEADataSource requires q.entities registry keys")

        rows = []
        for entity in entities:
            config = self._registry.get(str(entity))
            if config is None:
                continue
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
