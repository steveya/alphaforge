from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .parsing import parse_xlsx_bytes
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    make_entity_id,
    project_columns,
    to_float,
)


class ECWeeklyOilBulletinDataSource(DataSource):
    name = "ec_weekly_oil_bulletin"
    TABLE = "ec_oil_bulletin_weekly"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        bulletin_url: str = "https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en",
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._bulletin_url = bulletin_url

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "product", "country", "tax_flag"],
                entity_column="entity_id",
                time_column="date",
                native_freq="W",
            )
        }

    def _discover_links(self) -> list[tuple[str, str]]:
        payload = self._http.get_bytes(
            url=self._bulletin_url,
            source="ec_oil_bulletin",
            artifact_name="bulletin.html",
        )
        html = payload.decode(errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        out = []
        for href in hrefs:
            lower = href.lower()
            if lower.endswith((".xlsx", ".xls")):
                label = "WITH_TAX" if "tax" in lower else "NO_TAX"
                out.append((urljoin(self._bulletin_url, href), label))
        return out

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        rows = []
        for url, tax_flag in self._discover_links():
            payload = self._http.get_bytes(
                url=url,
                source="ec_oil_bulletin",
                artifact_name=Path(url.split("?")[0]).name,
            )
            frame = parse_xlsx_bytes(payload)
            frame.columns = [str(c).lower() for c in frame.columns]
            date_col = next(
                (c for c in frame.columns if "date" in c or "week" in c), None
            )
            product_col = next(
                (c for c in frame.columns if "product" in c or "fuel" in c), None
            )
            country_col = next(
                (c for c in frame.columns if "country" in c or c in {"ms", "geo"}), None
            )
            value_col = next(
                (c for c in frame.columns if "price" in c or "value" in c), None
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
            tmp["country"] = (
                frame[country_col].astype(str).str.upper() if country_col else "EU"
            )
            tmp["value"] = to_float(frame[value_col])
            tmp["tax_flag"] = tax_flag
            tmp["entity_id"] = [
                make_entity_id("ec_oil", prod, ctry, tax_flag)
                for prod, ctry in zip(tmp["product"], tmp["country"])
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
