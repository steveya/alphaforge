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
        self.fetch_many_calls = []

    def _result_for_query(self, query):
        entity = (
            query.entities[0]
            if query.entities
            else "cot.tff.eur.net"
        )
        return FetchResult(
            data=pd.DataFrame(
                {
                    "obs_date": pd.to_datetime(["2025-01-31"]),
                    "asof_utc": pd.to_datetime(["2025-02-07"]),
                    "value": [42.0],
                    "series_key": [entity],
                }
            ),
            source="cftc",
            dataset=query.table,
            is_pit=True,
            cached_at=None,
        )

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append((query, max_staleness))
        return self._result_for_query(query)

    def fetch_many(self, queries, *, max_staleness=None):
        self.fetch_many_calls.append((list(queries), max_staleness))
        return [self._result_for_query(query) for query in queries]

    def list_entities(self, dataset):
        return ["cot.tff.eur.net", "cot.tff.gbp.net", "cot.tff.jpy.net"]


class FakeBloombergAdapter(SourceAdapterBase):
    source_name = "bloomberg"
    datasets = frozenset({"cot.tff", "market.ohlcv"})

    def __init__(self):
        self.fetch_calls = []
        self.fetch_many_calls = []

    def _result_for_query(self, query):
        entity = query.entities[0] if query.entities else "cot.tff.eur.net"
        return FetchResult(
            data=pd.DataFrame(
                {
                    "obs_date": pd.to_datetime(["2025-01-31"]),
                    "value": [43.0],
                    "series_key": [entity],
                }
            ),
            source="bloomberg",
            dataset=query.table,
            is_pit=True,
            cached_at=None,
        )

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append((query, max_staleness))
        return self._result_for_query(query)

    def fetch_many(self, queries, *, max_staleness=None):
        self.fetch_many_calls.append((list(queries), max_staleness))
        return [self._result_for_query(query) for query in queries]

    def list_entities(self, dataset):
        if dataset == "cot.tff":
            return ["cot.tff.eur.net", "cot.tff.gbp.net"]
        return ["SPY", "AAPL"]


class FakeFREDAdapter(SourceAdapterBase):
    source_name = "fred"
    datasets = frozenset({"gdp"})

    def __init__(self):
        self.fetch_calls = []
        self.fetch_many_calls = []

    def _result_for_query(self, query):
        entity = query.entities[0] if query.entities else "GDP"
        return FetchResult(
            data=pd.DataFrame(
                {
                    "obs_date": pd.to_datetime(["2025-01-31"]),
                    "asof_utc": pd.to_datetime(["2025-02-15"]),
                    "value": [3.1],
                    "series_key": [entity],
                }
            ),
            source="fred",
            dataset=query.table,
            is_pit=True,
            cached_at=None,
        )

    def fetch(self, query, *, max_staleness=None):
        self.fetch_calls.append((query, max_staleness))
        return self._result_for_query(query)

    def fetch_many(self, queries, *, max_staleness=None):
        self.fetch_many_calls.append((list(queries), max_staleness))
        return [self._result_for_query(query) for query in queries]

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
    def test_from_adapters_derives_defaults_and_load_helper(self, cftc, fred):
        from alphaforge.data.context import DataContext

        ctx = DataContext.from_adapters(cftc, fred, calendars={}, store=None)

        assert ctx.default_sources == {"cot.tff": "cftc", "gdp": "fred"}

        result = ctx.load("gdp", columns=["value"], entities=["GDP"])

        assert result.source == "fred"
        assert len(fred.fetch_calls) == 1

    def test_from_adapters_requires_default_for_ambiguous_datasets(self, cftc, bbg):
        from alphaforge.data.context import DataContext

        ctx = DataContext.from_adapters(cftc, bbg, calendars={}, store=None)

        with pytest.raises(KeyError, match="Multiple adapters serve dataset 'cot.tff'"):
            ctx.load("cot.tff", columns=["value"], entities=["cot.tff.eur.net"])

    def test_fetch_routes_to_adapter(self, ctx, fred):
        q = Query(table="gdp", columns=["value"], entities=["GDP"])
        result = ctx.fetch(q)
        assert result.source == "fred"
        assert len(fred.fetch_calls) == 1

    def test_fetch_many_batches_by_adapter_and_preserves_input_order(
        self, ctx, cftc, fred
    ):
        queries = [
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"]),
            Query(table="gdp", columns=["value"], entities=["GDP"]),
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.jpy.net"]),
        ]
        results = ctx.fetch_many(queries)
        assert len(results) == 3
        assert [result.source for result in results] == ["cftc", "fred", "cftc"]
        assert [result.data["series_key"].iloc[0] for result in results] == [
            "cot.tff.eur.net",
            "GDP",
            "cot.tff.jpy.net",
        ]
        assert len(cftc.fetch_many_calls) == 1
        assert len(fred.fetch_many_calls) == 1
        assert len(cftc.fetch_calls) == 0
        assert len(fred.fetch_calls) == 0

    def test_fetch_many_with_explicit_source_uses_single_adapter_batch(
        self, ctx, bbg, cftc
    ):
        queries = [
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"]),
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.gbp.net"]),
        ]
        results = ctx.fetch_many(queries, source="bloomberg")
        assert [result.source for result in results] == ["bloomberg", "bloomberg"]
        assert len(bbg.fetch_many_calls) == 1
        assert len(bbg.fetch_calls) == 0
        assert len(cftc.fetch_many_calls) == 0

    def test_fetch_many_forwards_max_staleness_to_batch_fetches(
        self, ctx, cftc, fred
    ):
        threshold = pd.Timedelta(days=2)
        queries = [
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"]),
            Query(table="gdp", columns=["value"], entities=["GDP"]),
        ]
        ctx.fetch_many(queries, max_staleness=threshold)
        assert cftc.fetch_many_calls[0][1] == threshold
        assert fred.fetch_many_calls[0][1] == threshold

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

    def test_canonical_fetch_does_not_fallback_to_legacy_sources(self, tmp_path, cftc):
        from alphaforge.data.context import DataContext
        from alphaforge.store.duckdb_parquet import DuckDBParquetStore

        class LegacyOnlySource:
            name = "cftc"

            def __init__(self):
                self.fetch_calls = []

            def schemas(self):
                return {}

            def fetch(self, q):
                self.fetch_calls.append(q)
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2025-01-31"]),
                        "entity_id": ["legacy.entity"],
                        "value": [999.0],
                    }
                )

        legacy = LegacyOnlySource()
        ctx = DataContext(
            sources={"cftc": legacy},
            calendars={},
            store=DuckDBParquetStore(root=str(tmp_path / "store")),
            adapters={"cftc": cftc},
            default_sources={"cot.tff": "cftc"},
        )

        result = ctx.fetch(
            Query(table="cot.tff", columns=["value"], entities=["cot.tff.eur.net"])
        )

        assert result.source == "cftc"
        assert len(cftc.fetch_calls) == 1
        assert len(legacy.fetch_calls) == 0

    def test_canonical_fetch_requires_adapter_registration(self, tmp_path):
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

        ctx = DataContext(
            sources={"dummy": DummySource(ohlcv, macro)},
            calendars={},
            store=MemoryStore(),
        )

        with pytest.raises(KeyError, match="No adapters registered"):
            ctx.fetch(Query(table="market.ohlcv", columns=["close"], entities=["AAA"]))

    def test_pit_accessor_still_works(self, ctx):
        """ctx.pit should still be a PITAccessor (when DuckDB store)."""
        from alphaforge.pit.accessor import PITAccessor

        assert isinstance(ctx.pit, PITAccessor)
