from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .schema_helpers import table_schema
from .tabular import TabularDocumentSourceBase, artifact_name_from_url
from .utils import ensure_date_utc, make_entity_id, to_float


class EzoicAdRevenueDailySource(TabularDocumentSourceBase):
    name: str = "ezoic_adrevenue_daily"
    TABLE = "ezoic.adrevenue.daily"
    URL = "https://adrevenueindex.ezoic.com/"

    def __init__(
        self,
        *,
        page_url: str | None = None,
        data_url: str | None = None,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._page_url = page_url or self.URL
        self._data_url = data_url
        super().__init__(http_client=http_client, cache_dir=cache_dir)

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "region", "category"],
                native_freq="D",
                time_semantics="point",
            )
        }

    @staticmethod
    def _coerce_records(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        return [row for row in payload if isinstance(row, dict)]

    def _extract_records_from_html(self, html: str) -> list[dict[str, Any]]:
        patterns = [
            r"window\.__INITIAL_DATA__\s*=\s*(\{.*?\});",
            r"series\s*:\s*(\[\{.*?\}\])",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.DOTALL)
            if not match:
                continue
            blob = match.group(1)
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, list):
                    return self._coerce_records(parsed)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("series"), list):
                        return self._coerce_records(parsed["series"])
                    if isinstance(parsed.get("data"), list):
                        return self._coerce_records(parsed["data"])
            except json.JSONDecodeError:
                continue
        return []

    def _load_records(self) -> list[dict[str, Any]]:
        if self._data_url:
            payload = self._http.get_bytes(
                url=self._data_url,
                source="ezoic_adrevenue_daily",
                artifact_name=artifact_name_from_url(self._data_url, "adrevenue.json"),
            )
            parsed = json.loads(payload.decode())
            if isinstance(parsed, list):
                return self._coerce_records(parsed)
            if isinstance(parsed, dict):
                records = parsed.get("data", parsed.get("series", []))
                return self._coerce_records(records)
            return []

        payload = self._http.get_bytes(
            url=self._page_url,
            source="ezoic_adrevenue_daily",
            artifact_name=artifact_name_from_url(self._page_url, "adrevenue.html"),
        )
        html = payload.decode(errors="ignore")
        return self._extract_records_from_html(html)

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)

        records = self._load_records()
        schema = self._schema()
        if not records:
            return self._empty_frame(schema)

        frame = pd.DataFrame(records)
        date_col = "date" if "date" in frame.columns else "day"
        value_col = "value" if "value" in frame.columns else "index"
        region_col = "region" if "region" in frame.columns else None
        category_col = "category" if "category" in frame.columns else None

        out = pd.DataFrame(index=frame.index)
        out["date"] = ensure_date_utc(frame[date_col])
        out["value"] = to_float(frame[value_col])
        out["region"] = (
            frame[region_col].astype(str).str.lower() if region_col else "global"
        )
        out["category"] = (
            frame[category_col].astype(str).str.lower() if category_col else "all"
        )
        out["asof_utc"] = self._asof_utc(q)
        out["entity_id"] = [
            make_entity_id("macro", "index", "adrevenue", region, "value", "ezoic")
            for region in out["region"]
        ]

        return self._finalize(out, q=q, schema=schema, sort_by=[])
