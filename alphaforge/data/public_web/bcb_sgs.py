from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import apply_query_filters, make_entity_id, project_columns


class BCBSGSDataSource(DataSource):
    name = "bcb_sgs"
    TABLE = "bcb_sgs_series"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://api.bcb.gov.br/dados/serie/bcdata.sgs",
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")

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

    def _call(self, code: str, q: Query) -> pd.DataFrame:
        params = {"formato": "csv"}
        if q.start is not None:
            params["dataInicial"] = q.start.strftime("%d/%m/%Y")
        if q.end is not None:
            params["dataFinal"] = q.end.strftime("%d/%m/%Y")
        url = f"{self._base_url}.{code}/dados?{urlencode(params)}"
        payload = self._http.get_bytes(
            url=url, source="bcb_sgs", artifact_name=f"{code}.csv"
        )
        frame = pd.read_csv(io.StringIO(payload.decode("utf-8", errors="ignore")))
        frame.columns = [str(c).lower() for c in frame.columns]
        return frame

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        codes = [str(x) for x in (q.entities or [])]
        if not codes:
            raise ValueError("BCBSGSDataSource requires q.entities with SGS codes")

        rows = []
        for code in codes:
            df = self._call(code, q)
            if "data" not in df.columns or "valor" not in df.columns:
                continue
            dates = pd.to_datetime(
                df["data"], format="%d/%m/%Y", errors="coerce", utc=True
            )
            vals = pd.to_numeric(
                df["valor"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            tmp = pd.DataFrame(
                {
                    "date": dates,
                    "entity_id": make_entity_id(code),
                    "value": vals,
                    "asof_utc": q.asof or pd.Timestamp.now(tz="UTC"),
                }
            )
            rows.append(tmp)

        out = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["date", "entity_id", "asof_utc", "value"])
        )
        if out.empty:
            return out
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
