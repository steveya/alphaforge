"""Bank of England Real-Time Database adapter for UK data."""

from __future__ import annotations

from datetime import date
from typing import Any

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.exceptions import PITNotSupportedError
from alphaforge.pit.observation import PITObservation, SeriesMetadata


class BOERTDBAdapter(PITAdapter):
    """Adapter for Bank of England Real-Time Database (UK GDP).

    Stub implementation — to be completed with actual BoE RTDB integration.
    """

    @property
    def name(self) -> str:
        return "BOE_RTDB"

    def supports_pit(self, series_id: str) -> bool:
        return False

    def list_vintages(self, series_id: str) -> list[date]:
        raise PITNotSupportedError(series_id, "BoE RTDB adapter not yet implemented")

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
        raise PITNotSupportedError(series_id, "BoE RTDB adapter not yet implemented")
