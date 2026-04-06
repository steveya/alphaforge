from __future__ import annotations

from pathlib import Path
from typing import Any

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .registry_loader import load_registry_entries, map_registry


class RegistryApiSourceBase(PublicWebSourceBase):
    def _init_registry(
        self,
        file_name: str,
        *,
        registry_entries: list[dict[str, Any]] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        entries = load_registry_entries(
            file_name,
            entries=registry_entries,
            registry_path=registry_path,
        )
        self._registry = map_registry(entries)

    def _iter_entity_configs(
        self,
        q: Query,
        *,
        error_message: str,
    ):
        for entity in self._require_entities(q, error_message=error_message):
            config = self._registry.get(str(entity))
            if config is None:
                continue
            yield str(entity), config
