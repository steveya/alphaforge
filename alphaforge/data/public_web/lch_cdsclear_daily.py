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


class LCHCDSClearDailySource(TabularDocumentSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "metric", "segment"],
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
            url=self._volumes_url,
            source="lch_cdsclear_daily",
            artifact_name=Path(self._volumes_url.split("?")[0]).name or "volumes.html",
        )

        rows: list[pd.DataFrame] = []
        for table in candidate_tables(
            tables,
            any_of=("value", "volume", "notional", "trades"),
        ):
            out = pd.DataFrame(index=table.index)
            out["date"] = resolved_date_series(
                table,
                ["date", "trading_day"],
                default_date=snapshot_date,
            )
            out["metric"] = resolved_text_series(
                table,
                ["metric"],
                default="volume",
                case="lower",
                space_replacement="_",
            )
            out["segment"] = resolved_text_series(
                table,
                ["segment", "family", "index_family"],
                default="all",
                case="lower",
                space_replacement="_",
            )
            out["value"] = resolved_numeric_series(
                table,
                ["value", "volume", "notional", "trades"],
            )

            out["entity_id"] = [
                make_entity_id("credit", "cds", seg, metric, "lch")
                for seg, metric in zip(out["segment"], out["metric"])
            ]
            out["asof_utc"] = asof_utc
            rows.append(out)

        out = pd.concat(rows, ignore_index=True) if rows else self._empty_frame(schema)
        return self._finalize(out, q=q, schema=schema, sort_by=[])
