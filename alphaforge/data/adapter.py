"""Unified SourceAdapter protocol and base mixin.

SourceAdapter   — Protocol that all data-source adapters must satisfy.
SourceAdapterBase — Mixin providing default ``fetch_many`` and ``prefetch``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Protocol, runtime_checkable

from .contracts import DatasetContract, FetchPolicy, UnsupportedOperationError
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

    def describe(self, dataset: str) -> DatasetContract:
        """Declare the schema, query support and temporal guarantees."""
        ...

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
        policy: FetchPolicy | None = None,
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
        policy: FetchPolicy | None = None,
    ) -> list[FetchResult]:
        """Batch fetch for multiple queries.

        Bulk adapters can optimize this (single cache lookup for all keys).
        The default implementation fetches queries sequentially.
        """
        ...

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Explicitly warm cache for a dataset.

        Unsupported adapters raise UnsupportedOperationError.
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
    Unsupported optional operations raise ``UnsupportedOperationError``.
    """

    source_name: str
    datasets: frozenset[str]

    def describe(self, dataset: str) -> DatasetContract:
        """Legacy/custom subclasses can add descriptors incrementally."""
        if dataset not in self.datasets:
            raise KeyError(f"Unknown dataset: {dataset!r}")
        raise UnsupportedOperationError(f"{self.source_name} has no dataset descriptor")

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
        policy: FetchPolicy | None = None,
    ) -> FetchResult:
        raise NotImplementedError

    def fetch_many(
        self,
        queries: list[Query],
        *,
        max_staleness: Optional[timedelta] = None,
        policy: FetchPolicy | None = None,
    ) -> list[FetchResult]:
        """Default: iterate and call ``fetch`` for each query."""
        if policy is None:
            # Preserve old third-party fetch signatures by default.
            return [self.fetch(q, max_staleness=max_staleness) for q in queries]
        return [self.fetch(q, max_staleness=max_staleness, policy=policy) for q in queries]

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Unsupported warming must not look like an empty acquisition."""
        raise UnsupportedOperationError(f"{self.source_name} does not support prefetch")

    def list_entities(self, dataset: str) -> list[str]:
        raise UnsupportedOperationError(f"{self.source_name} does not support entity discovery")
