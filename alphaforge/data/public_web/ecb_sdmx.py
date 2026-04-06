from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .registry_api import RegistryApiSourceBase
from .schema_helpers import single_value_schema


class ECBSDMXDataSource(RegistryApiSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        self._init_registry(
            "ecb_sdmx_series.yaml",
            registry_entries=registry_entries,
            registry_path=registry_path,
        )

    def schemas(self):
        return {
            self.TABLE: single_value_schema(self.TABLE, native_freq="M")
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
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for entity, cfg in self._iter_entity_configs(
            q,
            error_message="ECBSDMXDataSource requires q.entities registry keys",
        ):
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
                        "asof_utc": asof_utc,
                    }
                )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
