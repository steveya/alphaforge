"""Unified SourceAdapter protocol and base mixin.

SourceAdapter   — Protocol that all data-source adapters must satisfy.
SourceAdapterBase — Mixin providing default ``fetch_many`` and ``prefetch``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional, Protocol, runtime_checkable

from .query import Query
from .types import CacheManifest, FetchResult


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol for unified data-source adapters.

    Each adapter represents one *source* (e.g. ``"cftc"``, ``"fred"``,
    ``"tiingo"``) that can serve one or more *datasets*.

    Attributes
    ----------
    source_name : str
        Identifier for this source (e.g. ``"cftc"``, ``"bloomberg"``).
    datasets : frozenset[str]
        Logical datasets this adapter can serve (e.g. ``{"cot.tff", "cot.disagg"}``).
    """

    source_name: str
    datasets: frozenset[str]

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Cache-aware single query fetch.

        Checks local cache first; on miss, fetches from the underlying
        source, transforms to canonical format, persists to cache, and
        returns the result.
        """
        ...

    def fetch_many(
        self,
        queries: list[Query],
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> list[FetchResult]:
        """Batch fetch for multiple queries.

        Bulk adapters can optimize this (single cache lookup for all keys).
        Per-series adapters parallelize individual ``fetch`` calls.
        """
        ...

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Explicitly warm cache for a dataset.

        Bulk adapters fetch the full dataset; per-series adapters may no-op.
        Returns a manifest describing what was cached.
        """
        ...

    def list_entities(self, dataset: str) -> list[str]:
        """Return available entity/series keys for a dataset."""
        ...


class SourceAdapterBase:
    """Mixin providing default implementations for ``SourceAdapter``.

    Subclasses must define:
    - ``source_name``
    - ``datasets``
    - ``fetch()``
    - ``list_entities()``

    Default ``fetch_many`` iterates over queries calling ``fetch``.
    Default ``prefetch`` returns an empty manifest.
    """

    source_name: str
    datasets: frozenset[str]

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        raise NotImplementedError

    def fetch_many(
        self,
        queries: list[Query],
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> list[FetchResult]:
        """Default: iterate and call ``fetch`` for each query."""
        return [self.fetch(q, max_staleness=max_staleness) for q in queries]

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Default: no-op, returns empty manifest."""
        return CacheManifest(
            dataset=dataset,
            source=self.source_name,
            entity_keys=[],
            asof_range=asof_range or (date.min, date.min),
            populated_at=datetime.now(timezone.utc),
            row_count=0,
        )

    def list_entities(self, dataset: str) -> list[str]:
        raise NotImplementedError
