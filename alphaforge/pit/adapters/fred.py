"""FRED/ALFRED adapter for US macroeconomic data.

Requires ``requests`` (install via ``pip install alphaforge[fred]``).
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from typing import Any

import requests

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.pit.exceptions import PITNotSupportedError, SourceFetchError, VintageNotFoundError
from alphaforge.pit.observation import PITObservation, SeriesMetadata
from alphaforge.pit.vintage import select_vintage_for_asof


class FREDALFREDAdapter(PITAdapter):
    """Adapter for FRED/ALFRED (Federal Reserve Economic Data).

    Uses the FRED API to retrieve point-in-time data using realtime_start/realtime_end.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize FRED adapter.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        """
        self._api_key = api_key or os.getenv("FRED_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._session = requests.Session()
        self._vintage_cache: dict[str, tuple[list[date], float]] = {}
        self._cache_ttl = 86400  # 24 hours

    @property
    def name(self) -> str:
        return "FRED_ALFRED"

    def supports_pit(self, series_id: str) -> bool:
        try:
            vintages = self.list_vintages(series_id)
            return len(vintages) > 0
        except Exception:
            return False

    def list_vintages(self, series_id: str) -> list[date]:
        cache_key = series_id
        if cache_key in self._vintage_cache:
            cached_data, cached_time = self._vintage_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data

        url = f"{self.BASE_URL}/series/vintagedates"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }

        try:
            response = self._request_with_retry(url, params)
            data = response.json()
            vintage_dates_str = data.get("vintage_dates", [])
            if not vintage_dates_str:
                vintages: list[date] = []
            else:
                vintages = [
                    datetime.strptime(v, "%Y-%m-%d").date() for v in vintage_dates_str
                ]
            self._vintage_cache[cache_key] = (vintages, time.time())
            return vintages
        except Exception as e:
            raise SourceFetchError("FRED_ALFRED", original_error=e) from e

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
        vintages = self.list_vintages(series_id)
        if not vintages:
            raise PITNotSupportedError(series_id, "No vintages available (non-revised series)")

        selected_vintage = select_vintage_for_asof(vintages, asof_date)
        if selected_vintage is None:
            raise VintageNotFoundError(series_id, asof_date)

        url = f"{self.BASE_URL}/series/observations"
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "realtime_start": asof_date.strftime("%Y-%m-%d"),
            "realtime_end": asof_date.strftime("%Y-%m-%d"),
        }
        if start:
            params["observation_start"] = start.strftime("%Y-%m-%d")
        if end:
            params["observation_end"] = end.strftime("%Y-%m-%d")

        try:
            response = self._request_with_retry(url, params)
            data = response.json()

            observations: list[PITObservation] = []
            for obs_dict in data.get("observations", []):
                value_str = obs_dict["value"]
                if value_str == ".":
                    continue
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue

                obs = PITObservation(
                    series_key=series_id,
                    source="FRED_ALFRED",
                    source_series_id=series_id,
                    asof_date=asof_date,
                    vintage_date=asof_date,
                    obs_date=datetime.strptime(obs_dict["date"], "%Y-%m-%d").date(),
                    value=value,
                    value_raw=value_str,
                    frequency=self._infer_frequency(obs_dict["date"]),
                    realtime_start=asof_date,
                    realtime_end=asof_date,
                    ingested_at=datetime.utcnow(),
                    provenance={"realtime_start": obs_dict.get("realtime_start")},
                )
                observations.append(obs)
            return observations
        except SourceFetchError:
            raise
        except Exception as e:
            raise SourceFetchError("FRED_ALFRED", original_error=e) from e

    def _request_with_retry(
        self, url: str, params: dict[str, str], max_retries: int = 3
    ) -> requests.Response:
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)
        raise SourceFetchError("FRED_ALFRED", "Max retries exceeded")

    def _infer_frequency(self, date_str: str) -> str:
        return "M"  # Default to monthly
