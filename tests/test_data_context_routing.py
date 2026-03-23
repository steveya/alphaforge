"""Phase 3 – DataContext extension: source resolution, routing, cache integration.

TDD: tests written BEFORE implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data.adapter import SourceAdapterBase
from alphaforge.data.query import Query
from alphaforge.data.types import FetchResult

# ---------------------------------------------------------------------------
# Helpers – fake adapters
# ---------------------------------------------------------------------------


class FakeCFTCAdapter(SourceAdapterBase):
    source_name = "cftc"
    datasets = frozenset({"cot.tff"})

    def __init__(self):
        self.fetch_calls = []

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append(query)
        df = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(["2025-01-31"]),
                "asof_utc": pd.to_datetime(["2025-02-07"]),
                "value": [42.0],
                "series_key": ["cot.tff.eur.net"],
            }
        )
        return FetchResult(
            data=df, source="cftc", dataset="cot.tff", is_pit=True, cached_at=None,
        )

    def list_entities(self, dataset):
        return ["cot.tff.eur.net", "cot.tff.gbp.net", "cot.tff.jpy.net"]


class FakeBloombergAdapter(SourceAdapterBase):
    source_name = "bloomberg"
    datasets = frozenset({"cot.tff", "market.ohlcv"})

    def __init__(self):
        self.fetch_calls = []

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append(query)
        df = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(["2025-01-31"]),
                "value": [43.0],
                "series_key": ["cot.tff.eur.net"],
            }
        )
        return FetchResult(
            data=df, source="bloomberg", dataset="cot.tff", is_pit=True, cached_at=None,
        )

    def list_entities(self, dataset):
        if dataset == "cot.tff":
            return ["cot.tff.eur.net", "cot.tff.gbp.net"]
        return ["SPY", "AAPL"]


class FakeFREDAdapter(SourceAdapterBase):
    source_name = "fred"
    datasets = frozenset({"gdp"})

    def __init__(self):
        self.fetch_calls = []

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append(query)
        df = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(["2025-01-31"]),
                "asof_utc": pd.to_datetime(["2025-02-15"]),
                "value": [3.1],
                "series_key": ["GDP"],
            }
        )
        return FetchResult(
            data=df, source="fred", dataset="gdp", is_pit=True, cached_at=None,
        )

    def list_entities(self, dataset):
        return ["GDP", "GDPC1"]


@pytest.fixture
def cftc():
    return FakeCFTCAdapter()


@pytest.fixture
def bbg():
    return FakeBloombergAdapter()


@pytest.fixture
def fred():
    return FakeFREDAdapter()


@pytest.fixture
def ctx(tmp_path, cftc, bbg, fred):
    """DataContext with three adapters and default source mapping."""
    from alphaforge.data.context import DataContext
    from alphaforge.store.duckdb_parquet import DuckDBParquetStore

    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    return DataContext(
        sources={},
        calendars={},
        store=store,
        adapters={"cftc": cftc, "bloomberg": bbg, "fred": fred},
        default_sources={"cot.tff": "cftc", "gdp": "fred"},
    )


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


class TestContextResolvesDefaultSource:
    def test_default_source_used(self, ctx, cftc):
        q = Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"])
        result = ctx.fetch(q)
        assert result.source == "cftc"
        assert len(cftc.fetch_calls) == 1

    def test_explicit_source_overrides(self, ctx, cftc, bbg):
        q = Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"])
        result = ctx.fetch(q, source="bloomberg")
        assert result.source == "bloomberg"
        assert len(bbg.fetch_calls) == 1
        assert len(cftc.fetch_calls) == 0

    def test_unknown_dataset_raises(self, ctx):
        q = Query(table="unknown.dataset", columns=["value"])
        with pytest.raises(KeyError, match="unknown.dataset"):
            ctx.fetch(q)

    def test_unknown_source_raises(self, ctx):
        q = Query(table="cot.tff", columns=["value"])
        with pytest.raises(KeyError, match="nonexistent"):
            ctx.fetch(q, source="nonexistent")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestContextRouting:
    def test_fetch_routes_to_adapter(self, ctx, fred):
        q = Query(table="gdp", columns=["value"], entities=["GDP"])
        result = ctx.fetch(q)
        assert result.source == "fred"
        assert len(fred.fetch_calls) == 1

    def test_fetch_many_delegates(self, ctx, cftc, fred):
        queries = [
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"]),
            Query(table="gdp", columns=["value"], entities=["GDP"]),
        ]
        results = ctx.fetch_many(queries)
        assert len(results) == 2
        assert len(cftc.fetch_calls) == 1
        assert len(fred.fetch_calls) == 1

    def test_prefetch_delegates(self, ctx, cftc):
        manifest = ctx.prefetch("cot.tff")
        assert manifest.source == "cftc"
        assert manifest.dataset == "cot.tff"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestContextBackwardCompat:
    def test_old_sources_dict_still_works(self, tmp_path):
        """Existing ctx.sources["dummy"].fetch() pattern must keep working."""
        from conftest import DummySource, MemoryStore

        ohlcv = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"]),
                "entity_id": ["AAA"],
                "close": [100.0],
            }
        )
        macro = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-31"]),
                "entity_id": ["CPI"],
                "value": [1.0],
            }
        )
        from alphaforge.data.context import DataContext

        store = MemoryStore()
        ctx = DataContext(
            sources={"dummy": DummySource(ohlcv, macro)},
            calendars={},
            store=store,
        )
        q = Query(table="market.ohlcv", columns=["close"], entities=["AAA"])
        df = ctx.sources["dummy"].fetch(q)
        assert len(df) == 1

    def test_pit_accessor_still_works(self, ctx):
        """ctx.pit should still be a PITAccessor (when DuckDB store)."""
        from alphaforge.pit.accessor import PITAccessor

        assert isinstance(ctx.pit, PITAccessor)
