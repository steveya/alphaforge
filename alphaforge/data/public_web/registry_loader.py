from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_registry_entries(
    file_name: str,
    *,
    entries: list[dict[str, Any]] | None = None,
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if entries is not None:
        return entries

    path = (
        Path(registry_path)
        if registry_path
        else Path(__file__).resolve().parents[1] / "registries" / file_name
    )
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Registry must be a list of entries: {path}")

    for entry in parsed:
        if not isinstance(entry, dict) or "entity_id" not in entry:
            raise ValueError(f"Each registry entry must include entity_id: {path}")
    return parsed


def map_registry(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(e["entity_id"]): e for e in entries}
