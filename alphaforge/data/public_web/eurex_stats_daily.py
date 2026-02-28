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


class EurexStatsDailySource(DataSource):
    name: str = "eurex_stats_daily"
    TABLE = "eurex.stats.daily"
    URL = "https://www.eurex.com/ex-en/data/statistics/market-statistics-online"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        stats_url: str | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._stats_url = stats_url or self.URL

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["volume", "open_interest"],
                canonical_columns=[
                    "volume",
                    "open_interest",
                    "product_group",
                    "product_name",
                    "contract_count",
                    "trades",
                ],
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
            url=self._stats_url,
            source="eurex_stats_daily",
            artifact_name=Path(self._stats_url.split("?")[0]).name
            or "market_stats.html",
        )

        tables = parse_html_tables(payload)
        rows: list[pd.DataFrame] = []
        for table in tables:
            cols = set(table.columns)
            if not ({"volume", "open_interest"} & cols):
                continue

            out = pd.DataFrame(index=table.index)
            date_col = first_existing(table, "date", "trading_day")
            out["date"] = (
                ensure_date_utc(table[date_col])
                if date_col
                else pd.Timestamp.now(tz="UTC").normalize()
            )

            pg_col = first_existing(table, "product_group", "group")
            pn_col = first_existing(table, "product_name", "product", "contract")
            out["product_group"] = (
                table[pg_col].astype(str).str.lower().str.replace(" ", "_", regex=False)
                if pg_col
                else "unknown"
            )
            out["product_name"] = (
                table[pn_col].astype(str).str.lower().str.replace(" ", "_", regex=False)
                if pn_col
                else "unknown"
            )

            vol_col = first_existing(table, "volume")
            oi_col = first_existing(table, "open_interest", "openinterest")
            out["volume"] = to_float(table[vol_col]) if vol_col else pd.NA
            out["open_interest"] = to_float(table[oi_col]) if oi_col else pd.NA

            trades_col = first_existing(table, "trades")
            contract_count_col = first_existing(table, "contract_count", "contracts")
            out["trades"] = to_float(table[trades_col]) if trades_col else pd.NA
            out["contract_count"] = (
                to_float(table[contract_count_col]) if contract_count_col else pd.NA
            )

            out["entity_id"] = [
                make_entity_id("eurex", group, name)
                for group, name in zip(out["product_group"], out["product_name"])
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
