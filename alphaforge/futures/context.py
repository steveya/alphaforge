"""Helpers for wiring futures artifacts into DataContext."""

from __future__ import annotations

from typing import Mapping

from ..data.context import DataContext
from ..store.store import Store
from ..time.calendar import TradingCalendar
from .adapter import DATASET_SPECS, FirstRateFuturesAdapter
from .config import FirstRateFuturesConfig


def build_first_rate_futures_context(
    config: FirstRateFuturesConfig,
    *,
    calendars: Mapping[str, TradingCalendar] | None = None,
    store: Store | None = None,
    source_name: str = "first_rate_futures",
) -> DataContext:
    adapter = FirstRateFuturesAdapter(config.artifact_root)
    return DataContext(
        sources={},
        calendars=dict(calendars or {}),
        store=store,
        adapters={source_name: adapter},
        default_sources={dataset: source_name for dataset in DATASET_SPECS},
    )
