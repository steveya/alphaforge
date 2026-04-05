from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .schema_helpers import single_value_schema
from .utils import make_entity_id


class BCBSGSDataSource(PublicWebSourceBase):
    name = "bcb_sgs"
    TABLE = "bcb_sgs_series"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://api.bcb.gov.br/dados/serie/bcdata.sgs",
    ) -> None:
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")

    def schemas(self):
        return {self.TABLE: single_value_schema(self.TABLE)}

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
        self._require_table(q)
        schema = self._schema()
        codes = self._require_entities(
            q,
            error_message="BCBSGSDataSource requires q.entities with SGS codes",
        )
        asof_utc = self._asof_utc(q)

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
                    "asof_utc": asof_utc,
                }
            )
            rows.append(tmp)

        out = (
            pd.concat(rows, ignore_index=True)
            if rows
            else self._empty_frame(schema)
        )
        return self._finalize(out, q=q, schema=schema)
