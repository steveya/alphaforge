"""Series catalog configuration management.

Parses YAML-based series metadata catalogs that describe macro series,
their PIT modes, and release rules.

Requires ``pyyaml`` (install via ``pip install alphaforge[catalog]``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from alphaforge.pit.observation import SeriesMetadata
from alphaforge.pit.release_rules import ReleaseRule


class SeriesCatalog:
    """Manages series metadata from configuration."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize catalog from configuration file.

        Args:
            config_path: Path to YAML configuration file. If None, starts empty.
        """
        self._metadata: dict[str, SeriesMetadata] = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: Path) -> None:
        """Load series metadata from YAML configuration."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not config:
            return

        for series_key, series_config in config.items():
            release_rule_raw = series_config.get("release_rule")
            release_rule = (
                ReleaseRule.from_dict(release_rule_raw)
                if release_rule_raw is not None
                else None
            )

            metadata = SeriesMetadata(
                series_key=series_key,
                country=series_config.get("country", ""),
                source=series_config.get("source", ""),
                source_series_id=series_config.get("source_series_id", ""),
                frequency=series_config.get("frequency", ""),
                pit_mode=series_config.get("pit_mode", "NO_PIT"),
                seasonal_adjustment=series_config.get("seasonal_adjustment"),
                units=series_config.get("units"),
                description=series_config.get("description"),
                transforms=series_config.get("transforms"),
                adapter=series_config.get("adapter"),
                obs_date_anchor=series_config.get("obs_date_anchor"),
                publication_lag_months=series_config.get("publication_lag_months"),
                release_rule=release_rule,
                blocks=series_config.get("blocks", {}),
            )
            self._metadata[series_key] = metadata

    def get(self, series_key: str) -> SeriesMetadata | None:
        """Get metadata for a series."""
        return self._metadata.get(series_key)

    def get_all(self) -> dict[str, SeriesMetadata]:
        """Get all series metadata."""
        return self._metadata.copy()

    def add(self, metadata: SeriesMetadata) -> None:
        """Add or update series metadata."""
        self._metadata[metadata.series_key] = metadata

    def list_series(
        self, country: str | None = None, source: str | None = None
    ) -> list[str]:
        """List series keys, optionally filtered.

        Args:
            country: Filter by country code.
            source: Filter by source name.

        Returns:
            Sorted list of series keys.
        """
        series = []
        for key, meta in self._metadata.items():
            if country and meta.country != country:
                continue
            if source and meta.source != source:
                continue
            series.append(key)
        return sorted(series)

    def supports_pit(self, series_key: str) -> bool:
        """Check if a series supports PIT retrieval."""
        meta = self.get(series_key)
        if not meta:
            return False
        return meta.pit_mode != "NO_PIT"
