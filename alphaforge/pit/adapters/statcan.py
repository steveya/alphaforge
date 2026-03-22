"""Statistics Canada Real-Time Database adapter for Canadian data."""

from __future__ import annotations

from datetime import date
from typing import Any

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.exceptions import PITNotSupportedError
from alphaforge.pit.observation import PITObservation, SeriesMetadata


class StatCanRealTimeAdapter(PITAdapter):
    """Adapter for Statistics Canada Real-Time Tables.

    Stub implementation — to be completed with actual StatCan integration.
    """

    @property
    def name(self) -> str:
        return "STATCAN_REALTIME"

    def supports_pit(self, series_id: str) -> bool:
        return False

    def list_vintages(self, series_id: str) -> list[date]:
        raise PITNotSupportedError(series_id, "StatCan adapter not yet implemented")

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
        raise PITNotSupportedError(series_id, "StatCan adapter not yet implemented")
