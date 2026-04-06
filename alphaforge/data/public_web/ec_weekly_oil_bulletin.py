from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .parsing import parse_xlsx_bytes
from .schema_helpers import table_schema
from .tabular import (
    TabularDocumentSourceBase,
    resolved_date_series,
    resolved_numeric_series,
    resolved_text_series,
)
from .utils import make_entity_id


class ECWeeklyOilBulletinDataSource(TabularDocumentSourceBase):
    name = "ec_weekly_oil_bulletin"
    TABLE = "ec_oil_bulletin_weekly"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        bulletin_url: str = "https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en",
    ) -> None:
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._bulletin_url = bulletin_url

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "product", "country", "tax_flag"],
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
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)
        snapshot_date = self._snapshot_date(q)

        rows = []
        for url, tax_flag in self._discover_links():
            payload = self._http.get_bytes(
                url=url,
                source="ec_oil_bulletin",
                artifact_name=Path(url.split("?")[0]).name,
            )
            frame = parse_xlsx_bytes(payload)
            value_columns = [c for c in frame.columns if "price" in c or "value" in c]
            if not value_columns:
                continue
            tmp = pd.DataFrame(index=frame.index)
            tmp["date"] = resolved_date_series(
                frame,
                [c for c in frame.columns if "date" in c or "week" in c],
                default_date=snapshot_date,
            )
            tmp["product"] = resolved_text_series(
                frame,
                [c for c in frame.columns if "product" in c or "fuel" in c],
                default="UNKNOWN",
                case="upper",
            )
            tmp["country"] = resolved_text_series(
                frame,
                [c for c in frame.columns if "country" in c or c in {"ms", "geo"}],
                default="EU",
                case="upper",
            )
            tmp["value"] = resolved_numeric_series(frame, value_columns)
            tmp["tax_flag"] = tax_flag
            tmp["entity_id"] = [
                make_entity_id("ec_oil", prod, ctry, tax_flag)
                for prod, ctry in zip(tmp["product"], tmp["country"])
            ]
            tmp["asof_utc"] = asof_utc
            rows.append(tmp)

        out = pd.concat(rows, ignore_index=True) if rows else self._empty_frame(schema)
        return self._finalize(out, q=q, schema=schema)
