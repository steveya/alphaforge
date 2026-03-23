"""Phase 5 – FREDAdapter: SourceAdapter wrapping FREDALFREDAdapter.

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
# Mock FRED API responses
# ---------------------------------------------------------------------------

VINTAGES_RESPONSE = {
    "vintage_dates": [
        "2025-01-30",
        "2025-02-28",
        "2025-03-28",
    ]
}

OBSERVATIONS_RESPONSE = {
    "observations": [
        {
            "date": "2024-10-01",
            "value": "23595.1",
            "realtime_start": "2025-01-30",
            "realtime_end": "2025-02-27",
        },
        {
            "date": "2024-07-01",
            "value": "23223.1",
            "realtime_start": "2025-01-30",
            "realtime_end": "2025-02-27",
        },
        {
            "date": "2024-04-01",
            "value": ".",
            "realtime_start": "2025-01-30",
            "realtime_end": "2025-02-27",
        },
    ]
}


class MockResponse:
    def __init__(self, data):
        self.status_code = 200
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _mock_request_with_retry(self_or_url, url_or_params=None, params=None, **kwargs):
    """Mock for FREDALFREDAdapter._request_with_retry(url, params)."""
    # Handle both bound (self, url, params) and unbound (url, params) calls
    if isinstance(self_or_url, str):
        actual_url = self_or_url
    else:
        actual_url = url_or_params
    if "vintagedates" in actual_url:
        return MockResponse(VINTAGES_RESPONSE)
    return MockResponse(OBSERVATIONS_RESPONSE)


@pytest.fixture
def adapter(tmp_path):
    from alphaforge.data.sources.fred import FREDSourceAdapter

    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    return FREDSourceAdapter(api_key="test_key", cache_conn=conn)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestFREDAdapterProtocol:
    def test_is_source_adapter(self, adapter):
        assert isinstance(adapter, SourceAdapter)

    def test_source_name(self, adapter):
        assert adapter.source_name == "fred"

    def test_datasets(self, adapter):
        # FRED serves many datasets — but at minimum marks itself for macro
        assert isinstance(adapter.datasets, frozenset)
        assert len(adapter.datasets) > 0


# ---------------------------------------------------------------------------
# Fetch with mocked HTTP
# ---------------------------------------------------------------------------


class TestFREDAdapterFetch:
    @patch("alphaforge.pit.adapters.fred.FREDALFREDAdapter._request_with_retry")
    def test_fetch_returns_pit_data(self, mock_req, adapter):
        mock_req.side_effect = _mock_request_with_retry
        q = Query(
            table="macro.fred",
            columns=["value"],
            entities=["GDP"],
            asof="2025-02-15",
        )
        result = adapter.fetch(q)

        from alphaforge.data.types import FetchResult

        assert isinstance(result, FetchResult)
        assert result.source == "fred"
        assert result.is_pit is True
        # Should have 2 obs (the "." missing value is skipped)
        assert len(result.data) == 2

    @patch("alphaforge.pit.adapters.fred.FREDALFREDAdapter._request_with_retry")
    def test_fetch_has_pit_columns(self, mock_req, adapter):
        mock_req.side_effect = _mock_request_with_retry
        q = Query(
            table="macro.fred",
            columns=["value"],
            entities=["GDP"],
            asof="2025-02-15",
        )
        result = adapter.fetch(q)
        assert "series_key" in result.data.columns
        assert "obs_date" in result.data.columns
        assert "asof_utc" in result.data.columns
        assert "value" in result.data.columns

    @patch("alphaforge.pit.adapters.fred.FREDALFREDAdapter._request_with_retry")
    def test_fetch_multiple_series(self, mock_req, adapter):
        mock_req.side_effect = _mock_request_with_retry
        q = Query(
            table="macro.fred",
            columns=["value"],
            entities=["GDP", "GDPC1"],
            asof="2025-02-15",
        )
        result = adapter.fetch(q)
        # Each series returns 2 valid obs → 4 total
        assert len(result.data) == 4
        assert set(result.data["series_key"].unique()) == {"GDP", "GDPC1"}


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestFREDAdapterCache:
    @patch("alphaforge.pit.adapters.fred.FREDALFREDAdapter._request_with_retry")
    def test_second_fetch_uses_cache(self, mock_req, adapter):
        mock_req.side_effect = _mock_request_with_retry
        q = Query(
            table="macro.fred",
            columns=["value"],
            entities=["GDP"],
            asof="2025-02-15",
        )
        adapter.fetch(q)
        initial_count = mock_req.call_count

        # Second fetch should use cache
        result2 = adapter.fetch(q)
        assert mock_req.call_count == initial_count
        assert result2.cached_at is not None

    @patch("alphaforge.pit.adapters.fred.FREDALFREDAdapter._request_with_retry")
    def test_max_staleness_zero_bypasses_cache(self, mock_req, adapter):
        mock_req.side_effect = _mock_request_with_retry
        q = Query(
            table="macro.fred",
            columns=["value"],
            entities=["GDP"],
            asof="2025-02-15",
        )
        adapter.fetch(q)
        initial_count = mock_req.call_count

        adapter.fetch(q, max_staleness=timedelta(0))
        assert mock_req.call_count > initial_count


# ---------------------------------------------------------------------------
# list_entities / prefetch
# ---------------------------------------------------------------------------


class TestFREDAdapterMisc:
    def test_list_entities_returns_list(self, adapter):
        entities = adapter.list_entities("macro.fred")
        assert isinstance(entities, list)

    def test_default_prefetch(self, adapter):
        manifest = adapter.prefetch("macro.fred")
        assert manifest.source == "fred"
