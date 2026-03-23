"""TiingoAdapter — SourceAdapter for Tiingo EOD market data.

Non-PIT adapter (market prices don't get revised). Fetches from the Tiingo
REST API and caches results in DuckDB via CacheLayer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
import requests

from ..adapter import SourceAdapterBase
from ..cache_layer import CacheLayer
from ..query import Query
from ..types import FetchResult

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.tiingo.com/tiingo/daily"

# Tiingo adjusted → canonical column mapping
_ADJ_MAP = {
    "adjOpen": "open",
    "adjHigh": "high",
    "adjLow": "low",
    "adjClose": "close",
    "adjVolume": "volume",
}


class TiingoAdapter(SourceAdapterBase):
    """Cache-aware adapter for Tiingo EOD market data.

    Parameters
    ----------
    api_key : str
        Tiingo API token.
    cache_conn : duckdb.DuckDBPyConnection | None
        DuckDB connection for caching. If None, no caching.
    use_adjusted : bool
        Use split/dividend-adjusted prices (default True).
    """

    source_name = "tiingo"
    datasets = frozenset({"market.ohlcv"})

    def __init__(
        self,
        api_key: str,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
        use_adjusted: bool = True,
    ) -> None:
        self._api_key = api_key
        self._use_adjusted = use_adjusted
        self._cache: CacheLayer | None = None
        if cache_conn is not None:
            self._cache = CacheLayer(cache_conn)

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Fetch OHLCV for requested tickers. Cache-aware."""
        entities = list(query.entities or [])
        if not entities:
            return FetchResult(
                data=pd.DataFrame(),
                source=self.source_name,
                dataset=query.table,
                is_pit=False,
                cached_at=None,
            )

        frames = []
        any_cached_at = None

        for ticker in entities:
            # Try cache first
            cached = self._cache_lookup(ticker, query, max_staleness)
            if cached is not None:
                df, cached_at = cached
                frames.append(df)
                any_cached_at = cached_at
                continue

            # Fetch from API
            df = self._fetch_ticker(ticker, query.start, query.end)
            if not df.empty:
                # Persist to cache
                self._cache_store(ticker, df, query)
            frames.append(df)

        if not frames:
            combined = pd.DataFrame()
        else:
            combined = pd.concat(frames, ignore_index=True)

        # Filter to requested columns
        keep_cols = ["series_key", "obs_date"]
        for col in (query.columns or []):
            if col in combined.columns:
                keep_cols.append(col)
        if combined.empty:
            combined = pd.DataFrame(columns=keep_cols)
        else:
            combined = combined[
                [c for c in keep_cols if c in combined.columns]
            ]

        return FetchResult(
            data=combined,
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=any_cached_at,
        )

    def list_entities(self, dataset: str) -> list[str]:
        """Tiingo has no fixed entity list."""
        return []

    # ------------------------------------------------------------------
    # Internal: API fetch
    # ------------------------------------------------------------------

    def _fetch_ticker(
        self,
        ticker: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch a single ticker from Tiingo API."""
        params: dict = {
            "token": self._api_key,
            "format": "json",
            "resampleFreq": "daily",
        }
        if start is not None:
            params["startDate"] = start.strftime("%Y-%m-%d")
        if end is not None:
            params["endDate"] = end.strftime("%Y-%m-%d")

        url = f"{_BASE_URL}/{ticker}/prices"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Normalize dates
        df["obs_date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.normalize()

        # Map to canonical columns
        if self._use_adjusted:
            # Drop unadjusted columns first, then rename adjusted
            unadj = [v for v in _ADJ_MAP.values() if v in df.columns]
            df = df.drop(columns=unadj, errors="ignore")
            renames = {k: v for k, v in _ADJ_MAP.items() if k in df.columns}
            df = df.rename(columns=renames)

        df["series_key"] = ticker
        canonical = ["series_key", "obs_date", "open", "high", "low", "close", "volume"]
        available = [c for c in canonical if c in df.columns]
        return df[available].copy()

    # ------------------------------------------------------------------
    # Internal: Cache
    # ------------------------------------------------------------------

    def _cache_lookup(
        self,
        ticker: str,
        query: Query,
        max_staleness: Optional[timedelta],
    ) -> Optional[tuple[pd.DataFrame, datetime]]:
        if self._cache is None:
            return None
        result = self._cache.lookup(
            series_key=ticker,
            dataset="market.ohlcv",
            source=self.source_name,
            is_pit=False,
            max_staleness=max_staleness,
        )
        if result is None:
            return None
        df, cached_at = result
        # Normalize obs_date to tz-naive for comparison (query timestamps are UTC)
        obs = pd.to_datetime(df["obs_date"]).dt.tz_localize(None)
        if query.start is not None:
            start_naive = query.start.tz_localize(None) if query.start.tzinfo else query.start
            df = df[obs >= start_naive]
            obs = obs[obs >= start_naive]
        if query.end is not None:
            end_naive = query.end.tz_localize(None) if query.end.tzinfo else query.end
            df = df[obs <= end_naive]
        return df, cached_at

    def _cache_store(
        self, ticker: str, df: pd.DataFrame, query: Query
    ) -> None:
        if self._cache is None:
            return
        # For market data, store "close" as "value" for the cache schema
        store_df = df[["series_key", "obs_date"]].copy()
        store_df["value"] = df["close"].values if "close" in df.columns else 0.0
        self._cache.store(
            store_df,
            dataset="market.ohlcv",
            source=self.source_name,
            is_pit=False,
        )
