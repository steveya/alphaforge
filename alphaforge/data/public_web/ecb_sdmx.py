from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .registry_loader import load_registry_entries, map_registry
from .utils import apply_query_filters, project_columns


class ECBSDMXDataSource(DataSource):
    name = "ecb_sdmx"
    TABLE = "ecb_sdmx_series"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://data-api.ecb.europa.eu/service/data",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        entries = load_registry_entries(
            "ecb_sdmx_series.yaml",
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
    def _parse_period(x: str) -> pd.Timestamp | None:
        txt = str(x)
        if len(txt) == 7 and txt[4] == "-":
            return pd.Timestamp(
                year=int(txt[:4]), month=int(txt[5:7]), day=1, tz="UTC"
            ) + pd.offsets.MonthEnd(0)
        return pd.to_datetime(txt, errors="coerce", utc=True)

    def _call(self, cfg: dict, q: Query) -> pd.DataFrame:
        flow_ref = str(cfg.get("flowRef"))
        key = str(cfg.get("key"))
        params = dict(cfg.get("params", {}))
        params.setdefault("format", "csvdata")
        if q.start is not None:
            params["startPeriod"] = q.start.strftime("%Y-%m")
        if q.end is not None:
            params["endPeriod"] = q.end.strftime("%Y-%m")
        url = f"{self._base_url}/{flow_ref}/{key}?{urlencode(params, doseq=True)}"
        payload = self._http.get_bytes(
            url=url,
            source="ecb_sdmx",
            artifact_name=f"{flow_ref}_{key.replace('.','_')}.csv",
        )
        frame = pd.read_csv(io.StringIO(payload.decode("utf-8", errors="ignore")))
        frame.columns = [str(c).lower() for c in frame.columns]
        return frame

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError("ECBSDMXDataSource requires q.entities registry keys")

        rows = []
        for entity in entities:
            cfg = self._registry.get(str(entity))
            if cfg is None:
                continue
            df = self._call(cfg, q)
            period_col = (
                "time_period"
                if "time_period" in df.columns
                else ("period" if "period" in df.columns else None)
            )
            value_col = (
                "obs_value"
                if "obs_value" in df.columns
                else ("value" if "value" in df.columns else None)
            )
            if period_col is None or value_col is None:
                continue
            for _, row in df.iterrows():
                date = self._parse_period(str(row[period_col]))
                if date is None or pd.isna(date):
                    continue
                rows.append(
                    {
                        "date": date,
                        "entity_id": str(entity),
                        "value": pd.to_numeric(row[value_col], errors="coerce"),
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
