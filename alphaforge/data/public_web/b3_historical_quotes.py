from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import apply_query_filters, project_columns


class B3HistoricalQuotesDataSource(DataSource):
    name = "b3_historical_quotes"
    TABLE = "b3_equity_quotes_daily"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        page_url: str = "https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/historical-data/equities/historical-quote-data/",
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._page_url = page_url

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["open", "high", "low", "close", "volume"],
                canonical_columns=["open", "high", "low", "close", "volume"],
                entity_column="ticker",
                time_column="date",
                native_freq="D",
            )
        }

    def _discover_links(self, q: Query) -> list[str]:
        payload = self._http.get_bytes(
            url=self._page_url, source="b3_quotes", artifact_name="landing.html"
        )
        html = payload.decode(errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        links = [
            urljoin(self._page_url, h)
            for h in hrefs
            if h.lower().endswith((".zip", ".txt", ".csv"))
        ]
        years = set()
        if q.start is not None:
            years.add(q.start.year)
        if q.end is not None:
            years.add(q.end.year)
        if years:
            links = [u for u in links if any(str(y) in u for y in years)] or links
        return sorted(set(links))

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
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        rows = []
        for link in self._discover_links(q):
            payload = self._http.get_bytes(
                url=link,
                source="b3_quotes",
                artifact_name=Path(link.split("?")[0]).name,
            )
            text = ""
            if link.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    names = [
                        n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))
                    ]
                    if not names:
                        continue
                    text = zf.read(names[0]).decode("latin-1", errors="ignore")
            else:
                text = payload.decode("latin-1", errors="ignore")
            parsed = self._parse_fixed_width(text)
            if not parsed.empty:
                parsed["asof_utc"] = q.asof or pd.Timestamp.now(tz="UTC")
                rows.append(parsed)

        out = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "asof_utc",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
        )
        if out.empty:
            return out
        out = apply_query_filters(
            out.rename(columns={"ticker": "entity_id"}),
            q=q,
            time_col="date",
            entity_col="entity_id",
        ).rename(columns={"entity_id": "ticker"})
        schema = self.schemas()[self.TABLE]
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col="date",
            entity_col="ticker",
        )
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)
