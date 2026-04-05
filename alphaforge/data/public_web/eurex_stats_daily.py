from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .schema_helpers import table_schema
from .tabular import (
    TabularDocumentSourceBase,
    candidate_tables,
    resolved_date_series,
    resolved_numeric_series,
    resolved_text_series,
)
from .utils import make_entity_id


class EurexStatsDailySource(TabularDocumentSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._stats_url = stats_url or self.URL

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["volume", "open_interest"],
                canonical_columns=[
                    "volume",
                    "open_interest",
                    "product_group",
                    "product_name",
                    "contract_count",
                    "trades",
                ],
                native_freq="D",
                time_semantics="point",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)
        snapshot_date = self._snapshot_date(q)

        tables = self._read_html_tables(
            url=self._stats_url,
            source="eurex_stats_daily",
            artifact_name=Path(self._stats_url.split("?")[0]).name or "market_stats.html",
        )
        rows: list[pd.DataFrame] = []
        for table in candidate_tables(tables, any_of=("volume", "open_interest")):
            out = pd.DataFrame(index=table.index)
            out["date"] = resolved_date_series(
                table,
                ["date", "trading_day"],
                default_date=snapshot_date,
            )
            out["product_group"] = resolved_text_series(
                table,
                ["product_group", "group"],
                default="unknown",
                case="lower",
                space_replacement="_",
            )
            out["product_name"] = resolved_text_series(
                table,
                ["product_name", "product", "contract"],
                default="unknown",
                case="lower",
                space_replacement="_",
            )
            out["volume"] = resolved_numeric_series(table, ["volume"])
            out["open_interest"] = resolved_numeric_series(
                table,
                ["open_interest", "openinterest"],
            )
            out["trades"] = resolved_numeric_series(table, ["trades"])
            out["contract_count"] = resolved_numeric_series(
                table,
                ["contract_count", "contracts"],
            )

            out["entity_id"] = [
                make_entity_id("eurex", group, name)
                for group, name in zip(out["product_group"], out["product_name"])
            ]
            out["asof_utc"] = asof_utc
            rows.append(out)

        out = pd.concat(rows, ignore_index=True) if rows else self._empty_frame(schema)
        return self._finalize(out, q=q, schema=schema, sort_by=[])
