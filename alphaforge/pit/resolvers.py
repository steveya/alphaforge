"""Vintage resolvers for point-in-time data projection.

A :class:`VintageResolver` translates a :class:`~alphaforge.pit.views.VintageView`
declaration into concrete ``asof_date`` resolution logic.  The resolver sits
between the caller (e.g. a backtest loop) and the PIT adapter: the caller asks
"what should I fetch for this (series, obs_date, asof_date) triple?", and the
resolver returns the *effective* asof_date to pass to the adapter.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from alphaforge.pit.views import VintageView

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VintageResolver(Protocol):
    """Resolves the effective asof_date for a fetch given a vintage view.

    Implementations are stateless (:class:`RealtimeResolver`,
    :class:`LatestResolver`) or carry an immutable pre-computed revision
    map (:class:`FrozenResolver`).
    """

    @property
    def view(self) -> VintageView:
        """The view this resolver implements."""
        ...

    def resolve(
        self,
        series_key: str,
        obs_date: date,
        requested_asof: date,
        has_pit: bool,
    ) -> date:
        """Return the effective asof_date the adapter should use.

        Args:
            series_key: Canonical series identifier.
            obs_date: Observation date being resolved.
            requested_asof: The walk-forward asof_date from the backtest.
            has_pit: Whether this series has vintage history.

        Returns:
            The asof_date to pass to ``adapter.fetch_asof()``.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete resolvers
# ---------------------------------------------------------------------------


class RealtimeResolver:
    """Identity projection — returns ``requested_asof`` unchanged.

    Stateless.  Zero overhead.
    """

    def __init__(self) -> None:
        self._view = VintageView.realtime()

    @property
    def view(self) -> VintageView:
        return self._view

    def resolve(
        self,
        series_key: str,
        obs_date: date,
        requested_asof: date,
        has_pit: bool,
    ) -> date:
        return requested_asof


class LatestResolver:
    """Collapses all vintages to the most recent available.

    Stateless.  Uses a far-future sentinel to exploit the adapter's
    existing "latest vintage ≤ asof" semantics.
    """

    _SENTINEL = date(2099, 12, 31)

    def __init__(self) -> None:
        self._view = VintageView.latest()

    @property
    def view(self) -> VintageView:
        return self._view

    def resolve(
        self,
        series_key: str,
        obs_date: date,
        requested_asof: date,
        has_pit: bool,
    ) -> date:
        return self._SENTINEL if has_pit else requested_asof


class FrozenResolver:
    """Resolves to the *n*-th release vintage for each (series, obs_date).

    Stateful: carries an immutable pre-computed revision map built at
    construction time.  After construction, :meth:`resolve` is a dict
    lookup with no adapter calls.

    The revision map is built *outside* this class (typically by the
    backtest harness which has access to the PIT adapter) and passed in
    as a plain ``dict``.  This keeps the resolver free of adapter
    dependencies.

    Args:
        revision_map: Mapping from ``(series_key, obs_date)`` to a
            **sorted** list of vintage dates on which that observation
            was revised.
        n_releases: Which release to freeze to (1-indexed).  Defaults to 3.
    """

    def __init__(
        self,
        revision_map: dict[tuple[str, date], list[date]],
        n_releases: int = 3,
    ) -> None:
        self._view = VintageView.frozen(n=n_releases)
        self._n = n_releases
        self._revision_map = revision_map

    @property
    def view(self) -> VintageView:
        return self._view

    def resolve(
        self,
        series_key: str,
        obs_date: date,
        requested_asof: date,
        has_pit: bool,
    ) -> date:
        if not has_pit:
            return requested_asof
        vintages = self._revision_map.get((series_key, obs_date))
        if vintages is None:
            return requested_asof  # no revision history → fall through
        idx = min(self._n, len(vintages)) - 1  # 0-indexed
        return vintages[idx]
