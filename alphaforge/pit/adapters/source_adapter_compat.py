"""Compatibility bridge: SourceAdapter → PITAdapter.

Wraps a unified ``SourceAdapter`` so it can be used anywhere the legacy
``PITAdapter`` interface is expected (e.g., in nowcast-data's PITDataManager).

This enables a gradual migration: callers keep using ``PITAdapter.fetch_asof()``,
while the underlying fetch goes through the unified data layer.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.observation import PITObservation, SeriesMetadata

if TYPE_CHECKING:
    from alphaforge.data.adapter import SourceAdapter


class SourceAdapterPITCompat(PITAdapter):
    """Bridge a ``SourceAdapter`` into the ``PITAdapter`` interface.

    Parameters
    ----------
    adapter : SourceAdapter
        The unified source adapter to wrap.
    dataset : str
        The dataset key to use in ``Query.table`` (e.g. ``"macro.fred"``).
    frequency : str
        Default frequency label for generated ``PITObservation`` records.
    """

    def __init__(
        self,
        adapter: SourceAdapter,
        dataset: str,
        frequency: str = "Q",
    ) -> None:
        self._adapter = adapter
        self._dataset = dataset
        self._frequency = frequency

    @property
    def name(self) -> str:
        return f"{self._adapter.source_name}_compat"

    def supports_pit(self, series_id: str) -> bool:
        """All series routed through this adapter are assumed PIT-capable."""
        return True

    def list_vintages(self, series_id: str) -> list[date]:
        """Not supported via SourceAdapter — return empty."""
        return []

    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: date | None = None,
        end: date | None = None,
        *,
        metadata: SeriesMetadata | None = None,
        **kwargs: Any,
    ) -> list[PITObservation]:
        """Fetch PIT observations by delegating to the unified SourceAdapter."""
        from alphaforge.data.query import Query

        query = Query(
            table=self._dataset,
            columns=["value"],
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
            entities=[series_id],
            asof=pd.Timestamp(asof_date),
        )

        result = self._adapter.fetch(query)

        if result.data.empty:
            return []

        observations: list[PITObservation] = []
        for _, row in result.data.iterrows():
            asof_val = row.get("asof_utc", asof_date)
            obs_date_val = row.get("obs_date", row.get("date"))
            value = row.get("value", 0.0)

            asof_d = (
                asof_val.date()
                if isinstance(asof_val, (pd.Timestamp, datetime))
                else asof_val
            )
            obs_d = (
                obs_date_val.date()
                if isinstance(obs_date_val, (pd.Timestamp, datetime))
                else obs_date_val
            )

            observations.append(
                PITObservation(
                    series_key=series_id,
                    source=self._adapter.source_name,
                    source_series_id=series_id,
                    asof_date=asof_d,
                    vintage_date=asof_d,
                    obs_date=obs_d,
                    value=float(value),
                    frequency=self._frequency,
                    ingested_at=datetime.now(timezone.utc),
                )
            )
        return observations
