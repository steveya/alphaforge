"""Concrete SourceAdapter implementations for various data vendors.

Use :func:`discover_adapters` to find all installed adapters via
``alphaforge.source_adapters`` entry points.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alphaforge.data.adapter import SourceAdapter

logger = logging.getLogger(__name__)


def discover_adapters() -> dict[str, type[SourceAdapter]]:
    """Discover installed SourceAdapter classes via entry points.

    Returns a mapping of ``{source_name: AdapterClass}`` for every package
    that registers an ``alphaforge.source_adapters`` entry point.

    Example
    -------
    >>> adapters = discover_adapters()
    >>> adapters.keys()
    dict_keys(['tiingo', 'fred', 'cftc', 'dtcc'])
    """
    import sys

    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points

        eps: list[Any] = list(entry_points(group="alphaforge.source_adapters"))
    else:
        # Python 3.10–3.11 compat
        from importlib.metadata import entry_points as _ep

        all_eps = _ep()
        if isinstance(all_eps, dict):
            eps = list(all_eps.get("alphaforge.source_adapters", []))
        else:
            eps = list(all_eps.select(group="alphaforge.source_adapters"))

    result: dict[str, type[SourceAdapter]] = {}
    for ep in eps:
        try:
            cls = ep.load()
            result[ep.name] = cls
        except Exception:
            logger.warning("Failed to load source adapter entry point %r", ep.name, exc_info=True)

    return result
