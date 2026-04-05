from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .registry_api import RegistryApiSourceBase
from .schema_helpers import single_value_schema


class IBGESidraDataSource(RegistryApiSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        self._init_registry(
            "ibge_sidra_series.yaml",
            registry_entries=registry_entries,
            registry_path=registry_path,
        )

    def schemas(self):
        return {
            self.TABLE: single_value_schema(self.TABLE, native_freq="M")
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
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for entity, cfg in self._iter_entity_configs(
            q,
            error_message="IBGESidraDataSource requires q.entities registry keys",
        ):
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
                        "asof_utc": asof_utc,
                    }
                )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
