from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import apply_query_filters, ensure_date_utc, make_entity_id, project_columns


class ANPFuelPricesDataSource(DataSource):
    name = "anp_fuel_prices"
    TABLE = "anp_fuel_prices_weekly"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        landing_url: str = "https://dados.gov.br/dados/conjuntos-dados/serie-historica-de-precos-de-combustiveis-e-de-glp",
        mode: str = "latest_only",
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._landing_url = landing_url
        self._mode = mode

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "product", "geo_level", "geo_code"],
                entity_column="entity_id",
                time_column="date",
                native_freq="W",
            )
        }

    def _discover_links(self) -> list[str]:
        payload = self._http.get_bytes(
            url=self._landing_url, source="anp_fuel", artifact_name="landing.html"
        )
        html = payload.decode(errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        links = [
            urljoin(self._landing_url, h)
            for h in hrefs
            if h.lower().endswith((".csv", ".zip"))
        ]
        links = sorted(set(links))
        if self._mode == "latest_only" and links:
            return [links[-1]]
        return links

    @staticmethod
    def _read_payload(url: str, payload: bytes) -> pd.DataFrame:
        if url.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    return pd.DataFrame()
                with zf.open(csvs[0]) as fh:
                    return pd.read_csv(fh)
        return pd.read_csv(io.StringIO(payload.decode("utf-8", errors="ignore")))

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        rows = []
        for link in self._discover_links():
            payload = self._http.get_bytes(
                url=link, source="anp_fuel", artifact_name=Path(link.split("?")[0]).name
            )
            frame = self._read_payload(link, payload)
            frame.columns = [str(c).lower() for c in frame.columns]
            date_col = next((c for c in frame.columns if "data" in c), None)
            product_col = next(
                (c for c in frame.columns if "produto" in c or "combust" in c), None
            )
            geo_col = next(
                (
                    c
                    for c in frame.columns
                    if "estado" in c or "uf" in c or "municipio" in c
                ),
                None,
            )
            value_col = next(
                (c for c in frame.columns if "preco" in c and "medio" in c), None
            )
            if value_col is None:
                continue
            tmp = pd.DataFrame(index=frame.index)
            tmp["date"] = (
                ensure_date_utc(frame[date_col])
                if date_col
                else pd.Timestamp.now(tz="UTC").normalize()
            )
            tmp["product"] = (
                frame[product_col].astype(str).str.upper() if product_col else "UNKNOWN"
            )
            tmp["geo_level"] = "STATE"
            tmp["geo_code"] = (
                frame[geo_col].astype(str).str.upper() if geo_col else "BR"
            )
            tmp["value"] = pd.to_numeric(
                frame[value_col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            tmp["entity_id"] = [
                make_entity_id("anp_fuel", p, gl, gc)
                for p, gl, gc in zip(tmp["product"], tmp["geo_level"], tmp["geo_code"])
            ]
            tmp["asof_utc"] = q.asof or pd.Timestamp.now(tz="UTC")
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
