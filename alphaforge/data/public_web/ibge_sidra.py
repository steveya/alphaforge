from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .registry_loader import load_registry_entries, map_registry
from .utils import apply_query_filters, project_columns


class IBGESidraDataSource(DataSource):
    name = "ibge_sidra"
    TABLE = "ibge_sidra_series"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://api.sidra.ibge.gov.br/values",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        entries = load_registry_entries(
            "ibge_sidra_series.yaml",
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

    def _call(self, cfg: dict) -> list[dict]:
        if "url" in cfg:
            url = str(cfg["url"])
        else:
            params = dict(cfg.get("params", {}))
            table = cfg.get("table", "")
            variable = cfg.get("variable", "")
            period = cfg.get("period", "all")
            geo = params.pop("geo", "1")
            path = f"/t/{table}/n{geo}/v/{variable}/p/{period}"
            qs = urlencode(params, doseq=True)
            url = f"{self._base_url}{path}" + (f"?{qs}" if qs else "")
        payload = self._http.get_bytes(
            url=url,
            source="ibge_sidra",
            artifact_name=f"{cfg.get('table','sidra')}.json",
        )
        parsed = json.loads(payload.decode("utf-8", errors="ignore"))
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _parse_period(value: str) -> pd.Timestamp | None:
        txt = str(value)
        if len(txt) == 6 and txt.isdigit():
            return pd.Timestamp(
                year=int(txt[:4]), month=int(txt[4:6]), day=1, tz="UTC"
            ) + pd.offsets.MonthEnd(0)
        return pd.to_datetime(txt, errors="coerce", utc=True)

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError("IBGESidraDataSource requires q.entities registry keys")

        rows = []
        for entity in entities:
            cfg = self._registry.get(str(entity))
            if cfg is None:
                continue
            data = self._call(cfg)
            for row in data:
                period = row.get("D3C") or row.get("Mês (Código)") or row.get("V")
                value = row.get("V") or row.get("Valor")
                date = self._parse_period(str(period))
                if date is None or pd.isna(date):
                    continue
                rows.append(
                    {
                        "date": date,
                        "entity_id": str(entity),
                        "value": pd.to_numeric(
                            str(value).replace(",", "."), errors="coerce"
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
