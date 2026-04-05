from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .archive import discover_archive_fetches, read_first_zip_member
from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .schema_helpers import table_schema


class B3HistoricalQuotesDataSource(PublicWebSourceBase):
    name = "b3_historical_quotes"
    TABLE = "b3_equity_quotes_daily"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        page_url: str = "https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/historical-data/equities/historical-quote-data/",
    ) -> None:
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._page_url = page_url

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["open", "high", "low", "close", "volume"],
                canonical_columns=["open", "high", "low", "close", "volume"],
                entity_column="ticker",
                time_column="date",
                native_freq="D",
            )
        }

    def _discover_links(self, q: Query):
        payload = self._http.get_bytes(
            url=self._page_url, source="b3_quotes", artifact_name="landing.html"
        )
        html = payload.decode(errors="ignore")
        years = set()
        if q.start is not None:
            years.add(q.start.year)
        if q.end is not None:
            years.add(q.end.year)
        return discover_archive_fetches(
            html,
            base_url=self._page_url,
            suffixes=(".zip", ".txt", ".csv"),
            years=years,
            fallback_artifact_prefix="b3_quotes",
        )

    @staticmethod
    def _parse_fixed_width(text: str) -> pd.DataFrame:
        # Minimal parser for COTAHIST-like fixed-width rows (layout can vary by year)
        rows = []
        for line in text.splitlines():
            if not line.startswith("01") or len(line) < 210:
                continue
            try:
                date = pd.to_datetime(line[2:10], format="%Y%m%d", utc=True)
                ticker = line[12:24].strip()
                open_px = float(line[56:69]) / 100
                high_px = float(line[69:82]) / 100
                low_px = float(line[82:95]) / 100
                close_px = float(line[108:121]) / 100
                volume = float(line[170:188])
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "open": open_px,
                        "high": high_px,
                        "low": low_px,
                        "close": close_px,
                        "volume": volume,
                    }
                )
            except Exception:
                continue
        return pd.DataFrame(rows)

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)
        schema = self._schema()
        asof_utc = self._asof_utc(q)

        rows = []
        for planned in self._discover_links(q):
            payload = self._http.get_bytes(
                url=planned.url,
                source="b3_quotes",
                artifact_name=planned.artifact_name,
            )
            text = ""
            if planned.url.lower().split("?", 1)[0].endswith(".zip"):
                member = read_first_zip_member(payload, suffixes=(".txt", ".csv"))
                if member is None:
                    continue
                _member_name, member_payload = member
                text = member_payload.decode("latin-1", errors="ignore")
            else:
                text = payload.decode("latin-1", errors="ignore")
            parsed = self._parse_fixed_width(text)
            if not parsed.empty:
                parsed["asof_utc"] = asof_utc
                rows.append(parsed)

        out = (
            pd.concat(rows, ignore_index=True)
            if rows
            else self._empty_frame(schema, entity_col="ticker")
        )
        return self._finalize(out, q=q, schema=schema, entity_col="ticker")
