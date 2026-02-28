from __future__ import annotations

import io
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


class EurostatDataSource(DataSource):
    name = "eurostat"
    TABLE = "eurostat_series"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        base_url: str = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
        registry_entries: list[dict] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._base_url = base_url.rstrip("/")
        entries = load_registry_entries(
            "eurostat_series.yaml",
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

    @staticmethod
    def _parse_period(period: str) -> pd.Timestamp | None:
        txt = (
            str(period).replace("M", "-")
            if "M" in str(period) and "-" not in str(period)
            else str(period)
        )
        try:
            if len(txt) == 7 and txt[4] == "-":
                year = int(txt[:4])
                month = int(txt[5:7])
                return pd.Timestamp(
                    year=year, month=month, day=1, tz="UTC"
                ) + pd.offsets.MonthEnd(0)
            return pd.to_datetime(txt, utc=True, errors="coerce")
        except Exception:
            return None

    def _call(self, cfg: dict) -> pd.DataFrame:
        dataset = str(cfg.get("dataset"))
        filters = dict(cfg.get("filters", {}))
        params = {**filters}
        url = f"{self._base_url}/{dataset}?{urlencode(params, doseq=True)}"
        payload = self._http.get_bytes(
            url=url, source="eurostat", artifact_name=f"{dataset}.json"
        )
        text = payload.decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(text)
            values = parsed.get("value", {})
            if isinstance(values, dict):
                rows = []
                # fallback extraction from JSON-stat-like sparse map using time labels when available
                time_labels = (
                    parsed.get("dimension", {})
                    .get("time", {})
                    .get("category", {})
                    .get("label", {})
                )
                for key, value in values.items():
                    period = time_labels.get(str(key), str(key))
                    rows.append({"period": period, "value": value})
                return pd.DataFrame(rows)
        except Exception:
            pass
        # fallback CSV parse
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [str(c).lower() for c in frame.columns]
        return frame

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")
        entities = list(q.entities or [])
        if not entities:
            raise ValueError("EurostatDataSource requires q.entities registry keys")

        rows = []
        for entity in entities:
            cfg = self._registry.get(str(entity))
            if cfg is None:
                continue
            df = self._call(cfg)
            period_col = (
                "period"
                if "period" in df.columns
                else ("time" if "time" in df.columns else None)
            )
            value_col = (
                "value"
                if "value" in df.columns
                else ("obs_value" if "obs_value" in df.columns else None)
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
