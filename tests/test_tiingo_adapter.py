"""Phase 4 – TiingoAdapter: SourceAdapterBase for Tiingo EOD market data.

TDD: tests written BEFORE implementation.
Uses mocked HTTP to avoid real API calls.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import duckdb
import pytest

from alphaforge.data.adapter import SourceAdapter
from alphaforge.data.query import Query

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TIINGO_RESPONSE = [
    {
        "date": "2025-01-02T00:00:00+00:00",
        "close": 480.0,
        "adjClose": 480.0,
        "high": 482.0,
        "adjHigh": 482.0,
        "low": 478.0,
        "adjLow": 478.0,
        "open": 479.0,
        "adjOpen": 479.0,
        "volume": 1000000,
        "adjVolume": 1000000,
    },
    {
        "date": "2025-01-03T00:00:00+00:00",
        "close": 482.5,
        "adjClose": 482.5,
        "high": 484.0,
        "adjHigh": 484.0,
        "low": 480.0,
        "adjLow": 480.0,
        "open": 481.0,
        "adjOpen": 481.0,
        "volume": 1100000,
        "adjVolume": 1100000,
    },
]


class MockResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._data


@pytest.fixture
def adapter(tmp_path):
    from alphaforge.data.sources.tiingo import TiingoAdapter

    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    return TiingoAdapter(api_key="test_key", cache_conn=conn)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestTiingoAdapterProtocol:
    def test_is_source_adapter(self, adapter):
        assert isinstance(adapter, SourceAdapter)

    def test_source_name(self, adapter):
        assert adapter.source_name == "tiingo"

    def test_datasets(self, adapter):
        assert "market.ohlcv" in adapter.datasets


# ---------------------------------------------------------------------------
# Fetch with mocked HTTP
# ---------------------------------------------------------------------------


class TestTiingoAdapterFetch:
    @patch("alphaforge.data.sources.tiingo.requests.get")
    def test_fetch_returns_fetch_result(self, mock_get, adapter):
        mock_get.return_value = MockResponse(TIINGO_RESPONSE)
        q = Query(
            table="market.ohlcv",
            columns=["close", "volume"],
            entities=["SPY"],
            start="2025-01-02",
            end="2025-01-03",
        )
        result = adapter.fetch(q)

        from alphaforge.data.types import FetchResult

        assert isinstance(result, FetchResult)
        assert result.source == "tiingo"
        assert result.dataset == "market.ohlcv"
        assert result.is_pit is False
        assert len(result.data) == 2

    @patch("alphaforge.data.sources.tiingo.requests.get")
    def test_fetch_includes_correct_columns(self, mock_get, adapter):
        mock_get.return_value = MockResponse(TIINGO_RESPONSE)
        q = Query(
            table="market.ohlcv",
            columns=["close", "volume"],
            entities=["SPY"],
            start="2025-01-02",
            end="2025-01-03",
        )
        result = adapter.fetch(q)
        assert "close" in result.data.columns
        assert "volume" in result.data.columns
        assert "series_key" in result.data.columns
        assert "obs_date" in result.data.columns

    @patch("alphaforge.data.sources.tiingo.requests.get")
    def test_fetch_multiple_tickers(self, mock_get, adapter):
        mock_get.return_value = MockResponse(TIINGO_RESPONSE)
        q = Query(
            table="market.ohlcv",
            columns=["close"],
            entities=["SPY", "AAPL"],
            start="2025-01-02",
            end="2025-01-03",
        )
        result = adapter.fetch(q)
        # Two tickers, each returning 2 rows
        assert len(result.data) == 4
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestTiingoAdapterCache:
    @patch("alphaforge.data.sources.tiingo.requests.get")
    def test_second_fetch_uses_cache(self, mock_get, adapter):
        mock_get.return_value = MockResponse(TIINGO_RESPONSE)
        q = Query(
            table="market.ohlcv",
            columns=["close"],
            entities=["SPY"],
            start="2025-01-02",
            end="2025-01-03",
        )
        # First fetch hits API
        adapter.fetch(q)
        assert mock_get.call_count == 1

        # Second fetch should use cache — no additional API call
        result2 = adapter.fetch(q)
        assert mock_get.call_count == 1
        assert result2.cached_at is not None
        assert len(result2.data) == 2

    @patch("alphaforge.data.sources.tiingo.requests.get")
    def test_max_staleness_zero_bypasses_cache(self, mock_get, adapter):
        mock_get.return_value = MockResponse(TIINGO_RESPONSE)
        q = Query(
            table="market.ohlcv",
            columns=["close"],
            entities=["SPY"],
            start="2025-01-02",
            end="2025-01-03",
        )
        adapter.fetch(q)
        assert mock_get.call_count == 1

        # Force fresh fetch
        adapter.fetch(q, max_staleness=timedelta(0))
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# list_entities
# ---------------------------------------------------------------------------


class TestTiingoAdapterListEntities:
    def test_list_entities_not_supported(self, adapter):
        """Tiingo doesn't have a fixed entity list — returns empty."""
        entities = adapter.list_entities("market.ohlcv")
        assert entities == []
