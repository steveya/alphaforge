"""Phase 1 – SourceAdapter protocol and SourceAdapterBase mixin.

TDD: tests written BEFORE implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from alphaforge.data.query import Query

# ---------------------------------------------------------------------------
# Helpers – a minimal concrete adapter for testing
# ---------------------------------------------------------------------------


def _make_concrete_adapter():
    """Build a concrete adapter that extends SourceAdapterBase."""
    from alphaforge.data.adapter import SourceAdapterBase
    from alphaforge.data.types import FetchResult

    class FakeAdapter(SourceAdapterBase):
        source_name = "fake"
        datasets = frozenset({"test.dataset"})

        def __init__(self):
            self._fetch_count = 0

        def fetch(
            self,
            query: Query,
            *,
            max_staleness: Optional[timedelta] = None,
        ) -> FetchResult:
            self._fetch_count += 1
            df = pd.DataFrame(
                {
                    "obs_date": pd.to_datetime(["2025-01-31", "2025-02-28"]),
                    "value": [1.0, 2.0],
                }
            )
            return FetchResult(
                data=df,
                source=self.source_name,
                dataset=query.table,
                is_pit=False,
                cached_at=None,
            )

        def list_entities(self, dataset: str) -> list[str]:
            return ["entity_a", "entity_b", "entity_c"]

    return FakeAdapter()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestSourceAdapterProtocol:
    def test_is_runtime_checkable_protocol(self):
        from alphaforge.data.adapter import SourceAdapter

        assert hasattr(SourceAdapter, "__protocol_attrs__") or hasattr(
            SourceAdapter, "__abstractmethods__"
        )

    def test_concrete_satisfies_protocol(self):
        from alphaforge.data.adapter import SourceAdapter

        adapter = _make_concrete_adapter()
        assert isinstance(adapter, SourceAdapter)

    def test_plain_object_does_not_satisfy(self):
        from alphaforge.data.adapter import SourceAdapter

        class NotAnAdapter:
            pass

        assert not isinstance(NotAnAdapter(), SourceAdapter)


# ---------------------------------------------------------------------------
# SourceAdapterBase – default implementations
# ---------------------------------------------------------------------------


class TestSourceAdapterBaseAttributes:
    def test_source_name(self):
        adapter = _make_concrete_adapter()
        assert adapter.source_name == "fake"

    def test_datasets_frozenset(self):
        adapter = _make_concrete_adapter()
        assert isinstance(adapter.datasets, frozenset)
        assert "test.dataset" in adapter.datasets


class TestSourceAdapterBaseFetchMany:
    def test_fetch_many_delegates_to_fetch(self):
        adapter = _make_concrete_adapter()
        queries = [
            Query(table="test.dataset", columns=["value"], entities=["entity_a"]),
            Query(table="test.dataset", columns=["value"], entities=["entity_b"]),
        ]
        results = adapter.fetch_many(queries)
        assert len(results) == 2
        # Each query triggered a separate fetch call
        assert adapter._fetch_count == 2

    def test_fetch_many_keys_are_query_tables(self):
        """fetch_many returns a dict keyed by index (position)."""
        adapter = _make_concrete_adapter()
        queries = [
            Query(table="test.dataset", columns=["value"], entities=["entity_a"]),
        ]
        results = adapter.fetch_many(queries)
        assert len(results) == 1
        # Result is a list (ordered same as input)
        assert isinstance(results, list)

    def test_fetch_many_passes_max_staleness(self):
        """max_staleness is forwarded to individual fetch calls."""
        from alphaforge.data.adapter import SourceAdapterBase
        from alphaforge.data.types import FetchResult

        staleness_values = []

        class TrackingStaleness(SourceAdapterBase):
            source_name = "tracker"
            datasets = frozenset({"x"})

            def fetch(self, query, *, max_staleness=None):
                staleness_values.append(max_staleness)
                return FetchResult(
                    data=pd.DataFrame(),
                    source="tracker",
                    dataset="x",
                    is_pit=False,
                    cached_at=None,
                )

            def list_entities(self, dataset):
                return []

        adapter = TrackingStaleness()
        threshold = timedelta(hours=6)
        adapter.fetch_many(
            [Query(table="x", columns=[]), Query(table="x", columns=[])],
            max_staleness=threshold,
        )
        assert all(s == threshold for s in staleness_values)

    def test_fetch_many_empty_list(self):
        adapter = _make_concrete_adapter()
        results = adapter.fetch_many([])
        assert results == []


class TestSourceAdapterBasePrefetch:
    def test_default_prefetch_returns_empty_manifest(self):
        adapter = _make_concrete_adapter()
        manifest = adapter.prefetch("test.dataset")
        assert manifest.row_count == 0
        assert manifest.entity_keys == []
        assert manifest.source == "fake"
        assert manifest.dataset == "test.dataset"

    def test_default_prefetch_with_asof_range(self):
        adapter = _make_concrete_adapter()
        manifest = adapter.prefetch(
            "test.dataset",
            asof_range=(date(2025, 1, 1), date(2025, 3, 31)),
        )
        assert manifest.row_count == 0


class TestSourceAdapterBaseListEntities:
    def test_list_entities(self):
        adapter = _make_concrete_adapter()
        entities = adapter.list_entities("test.dataset")
        assert entities == ["entity_a", "entity_b", "entity_c"]
