from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    make_entity_id,
    project_columns,
    to_float,
)


class EzoicAdRevenueDailySource(DataSource):
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
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "region", "category"],
                entity_column="entity_id",
                time_column="date",
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
                artifact_name=Path(self._data_url.split("?")[0]).name
                or "adrevenue.json",
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
            artifact_name=Path(self._page_url.split("?")[0]).name or "adrevenue.html",
        )
        html = payload.decode(errors="ignore")
        return self._extract_records_from_html(html)

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        records = self._load_records()
        schema = self.schemas()[self.TABLE]
        if not records:
            return pd.DataFrame(
                columns=[
                    schema.time_column,
                    schema.entity_column,
                    "asof_utc",
                    *schema.required_columns,
                ]
            )

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
        out["asof_utc"] = pd.Timestamp.now(tz="UTC")
        out["entity_id"] = [
            make_entity_id("macro", "index", "adrevenue", region, "value", "ezoic")
            for region in out["region"]
        ]

        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
