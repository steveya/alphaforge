"""ECB Real-Time Database adapter for Euro Area data."""

from __future__ import annotations

from datetime import date
from typing import Any

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.exceptions import PITNotSupportedError
from alphaforge.pit.observation import PITObservation, SeriesMetadata


class ECBRTDBAdapter(PITAdapter):
    """Adapter for ECB Real-Time Database (Euro Area).

    Stub implementation — to be completed with actual ECB RTDB integration.
    """

    @property
    def name(self) -> str:
        return "ECB_RTDB"

    def supports_pit(self, series_id: str) -> bool:
        return False

    def list_vintages(self, series_id: str) -> list[date]:
        raise PITNotSupportedError(series_id, "ECB RTDB adapter not yet implemented")

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
        raise PITNotSupportedError(series_id, "ECB RTDB adapter not yet implemented")
