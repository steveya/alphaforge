"""Small, explicit contracts for provider acquisition and cache policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .query import Query

if TYPE_CHECKING:
    from .types import FetchResult

ResultKind = Literal["observations", "snapshot", "revisions"]
AvailabilityPrecision = Literal["none", "date", "timestamp"]
AvailabilityBasis = Literal["none", "release", "snapshot"]


class UnsupportedQueryError(ValueError):
    """A query requests semantics the dataset cannot supply."""


class UnsupportedOperationError(NotImplementedError):
    """An optional adapter operation is unavailable, rather than empty."""


class DataFetchError(RuntimeError):
    """A provider failed to acquire one requested entity."""

    def __init__(self, source: str, entity: str, message: str) -> None:
        self.source = source
        self.entity = entity
        super().__init__(f"{source}/{entity}: {message}")


@dataclass(frozen=True)
class FetchPolicy:
    """Acquisition policy; query semantics remain in :class:`Query`.

    ``reuse`` refreshes expired acquisitions, ``refresh`` always bypasses the
    cache, and ``allow_stale`` warns and returns expired acquisitions.
    Missing ``max_age`` means there is no age limit. Partial provider failures
    are errors unless explicitly enabled with ``allow_partial``.
    """

    cache_mode: Literal["reuse", "refresh", "allow_stale"] = "reuse"
    max_age: timedelta | None = None
    allow_partial: bool = False

    def __post_init__(self) -> None:
        if self.cache_mode not in {"reuse", "refresh", "allow_stale"}:
            raise ValueError(f"Unknown cache_mode: {self.cache_mode!r}")
        if self.max_age is not None and self.max_age < timedelta(0):
            raise ValueError("max_age must be nonnegative")


def resolve_fetch_policy(
    policy: FetchPolicy | None, max_staleness: timedelta | None
) -> FetchPolicy:
    """Keep the old warn-only threshold at an explicit compatibility boundary."""
    if policy is not None:
        if max_staleness is not None:
            raise ValueError("Cannot combine policy and max_staleness")
        return policy
    if max_staleness is None:
        return FetchPolicy()
    if max_staleness == timedelta(0):
        return FetchPolicy(cache_mode="refresh")
    return FetchPolicy(cache_mode="allow_stale", max_age=max_staleness)


@dataclass(frozen=True)
class DatasetContract:
    """Declared schema and temporal guarantees for one adapter dataset.

    Identity columns are always allowed and retained. Empty ``columns`` means
    all available columns; empty ``entities`` selects no rows, while ``None``
    requests the dataset's available universe where supported. ``revisions``
    describes rows with release availability, not completeness of all history.
    """

    columns: frozenset[str]
    result_kind: ResultKind
    availability_precision: AvailabilityPrecision = "none"
    availability_basis: AvailabilityBasis = "none"
    supports_asof: bool = False
    requires_entities: bool = False
    supports_prefetch: bool = False
    entity_discovery: Literal["none", "cached", "catalog"] = "none"

    @property
    def _identity_columns(self) -> frozenset[str]:
        columns = frozenset({"series_key", "obs_date"})
        if self.result_kind in {"snapshot", "revisions"}:
            columns = columns | {"asof_utc"}
        return columns

    def validate_query(self, query: Query) -> None:
        unknown = set(query.columns) - self.columns - self._identity_columns
        if unknown:
            raise UnsupportedQueryError(f"{query.table}: unsupported columns {sorted(unknown)}")
        if query.asof is not None and not self.supports_asof:
            raise UnsupportedQueryError(f"{query.table}: asof queries are unsupported")
        if query.vintage != "latest" or query.vintage_id is not None:
            raise UnsupportedQueryError(f"{query.table}: vintage selection is unsupported")
        if query.grid is not None:
            raise UnsupportedQueryError(f"{query.table}: grid alignment is unsupported")
        if self.requires_entities and query.entities is None:
            raise UnsupportedQueryError(f"{query.table}: an explicit entities selection is required")

    def validate_result(self, query: Query, result: FetchResult) -> None:
        """Defend declared guarantees at the context boundary, including empty frames."""
        if result.dataset != query.table:
            raise ValueError(f"Adapter returned dataset {result.dataset!r} for {query.table!r}")
        if result.result_kind != self.result_kind:
            raise ValueError("Adapter result_kind does not match its dataset contract")
        if result.is_pit != (self.result_kind in {"snapshot", "revisions"}):
            raise ValueError("Adapter is_pit does not match its dataset contract")
        if (result.availability_precision != self.availability_precision
                or result.availability_basis != self.availability_basis):
            raise ValueError("Adapter availability metadata does not match its dataset contract")
        snapshot_cutoff = None
        if self.result_kind == "snapshot":
            if result.snapshot_asof is None or pd.isna(result.snapshot_asof):
                raise ValueError("Selected snapshot results require a known snapshot_asof")
            snapshot_cutoff = pd.to_datetime(result.snapshot_asof, utc=True)
            if query.asof is not None:
                selected = snapshot_cutoff
                requested = query.asof
                if self.availability_precision == "date":
                    selected, requested = selected.normalize(), requested.normalize()
                if selected != requested:
                    raise ValueError("Selected snapshot does not match the requested asof")
        required = self._identity_columns | set(query.columns or self.columns)
        missing = required - set(result.data.columns)
        if missing:
            raise ValueError(f"Adapter result is missing columns {sorted(missing)}")
        unexpected = set(result.data.columns) - required
        if unexpected:
            raise ValueError(f"Adapter result contains unrequested columns {sorted(unexpected)}")
        if result.data.empty:
            return
        frame = result.data
        if frame[sorted(self._identity_columns)].isna().any().any():
            raise ValueError("Adapter result contains missing observation identities")
        if query.entities is not None and not frame["series_key"].isin(query.entities).all():
            raise ValueError("Adapter result contains unrequested entities")
        dates = pd.to_datetime(frame["obs_date"], utc=True)
        if ((query.start is not None and (dates < query.start).any())
                or (query.end is not None and (dates > query.end).any())):
            raise ValueError("Adapter result exceeds observation bounds")
        if snapshot_cutoff is not None:
            snapshot_available = pd.to_datetime(frame["asof_utc"], utc=True)
            if self.availability_precision == "date":
                snapshot_available = snapshot_available.dt.normalize()
                snapshot_cutoff = snapshot_cutoff.normalize()
            if (snapshot_available > snapshot_cutoff).any():
                raise ValueError("Adapter result exceeds the selected snapshot cutoff")
        if query.asof is not None and snapshot_cutoff is None:
            available = pd.to_datetime(frame["asof_utc"], utc=True)
            requested = query.asof
            if self.availability_precision == "date":
                available = available.dt.normalize()
                requested = requested.normalize()
            if (available > requested).any():
                raise ValueError("Adapter result exceeds the asof cutoff")
