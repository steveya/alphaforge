"""Generic entity registry mapping entity names to metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntityEntry:
    """Generic entity metadata entry."""

    entity_id: str
    asset_class: str
    source_keys: dict[str, str] = field(default_factory=dict)  # source → key pattern
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityRegistry:
    """Generic registry mapping entity names to metadata."""

    def __init__(self) -> None:
        self._entries: dict[str, EntityEntry] = {}

    def register(self, name: str, entry: EntityEntry) -> None:
        self._entries[name] = entry

    def get(self, name: str) -> EntityEntry:
        if name not in self._entries:
            raise KeyError(f"Unknown entity: {name!r}")
        return self._entries[name]

    def entities(self, asset_class: str | None = None) -> list[str]:
        if asset_class is None:
            return sorted(self._entries)
        return sorted(k for k, v in self._entries.items() if v.asset_class == asset_class)

    def source_key(self, entity: str, source: str) -> str:
        entry = self.get(entity)
        if source not in entry.source_keys:
            raise ValueError(f"Entity {entity!r} has no key for source {source!r}")
        return entry.source_keys[source]

    def all_source_keys(self, source: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, entry in self._entries.items():
            if source in entry.source_keys:
                result[name] = entry.source_keys[source]
        return result
