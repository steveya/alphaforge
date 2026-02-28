from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .parsing import parse_html_tables
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    first_existing,
    make_entity_id,
    project_columns,
    to_float,
)


class LCHCDSClearDailySource(DataSource):
    name: str = "lch_cdsclear_daily"
    TABLE = "lch.cdsclear.daily"
    URL = "https://www.lseg.com/en/post-trade/clearing/lch-services/cdsclear/volumes"

    def __init__(
        self,
        *,
        volumes_url: str | None = None,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._volumes_url = volumes_url or self.URL
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "metric", "segment"],
                entity_column="entity_id",
                time_column="date",
                native_freq="D",
                time_semantics="point",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        payload = self._http.get_bytes(
            url=self._volumes_url,
            source="lch_cdsclear_daily",
            artifact_name=Path(self._volumes_url.split("?")[0]).name or "volumes.html",
        )
        tables = parse_html_tables(payload)

        rows: list[pd.DataFrame] = []
        for table in tables:
            if not ({"value", "volume", "notional", "trades"} & set(table.columns)):
                continue

            out = pd.DataFrame(index=table.index)
            date_col = first_existing(table, "date", "trading_day")
            out["date"] = (
                ensure_date_utc(table[date_col])
                if date_col
                else pd.Timestamp.now(tz="UTC").normalize()
            )

            metric_col = first_existing(table, "metric")
            segment_col = first_existing(table, "segment", "family", "index_family")
            out["metric"] = (
                table[metric_col]
                .astype(str)
                .str.lower()
                .str.replace(" ", "_", regex=False)
                if metric_col
                else "volume"
            )
            out["segment"] = (
                table[segment_col]
                .astype(str)
                .str.lower()
                .str.replace(" ", "_", regex=False)
                if segment_col
                else "all"
            )

            value_col = first_existing(table, "value", "volume", "notional", "trades")
            out["value"] = to_float(table[value_col]) if value_col else pd.NA

            out["entity_id"] = [
                make_entity_id("credit", "cds", seg, metric, "lch")
                for seg, metric in zip(out["segment"], out["metric"])
            ]
            out["asof_utc"] = pd.Timestamp.now(tz="UTC")
            rows.append(out)

        schema = self.schemas()[self.TABLE]
        if not rows:
            return pd.DataFrame(
                columns=[
                    schema.time_column,
                    schema.entity_column,
                    "asof_utc",
                    *schema.required_columns,
                ]
            )

        out = pd.concat(rows, ignore_index=True)
        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
