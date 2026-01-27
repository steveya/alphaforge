from __future__ import annotations

from alphaforge.time.ref_period import RefPeriod


def make_ref_entity_id(series_key: str, ref: RefPeriod) -> str:
    if not series_key:
        raise ValueError("series_key is required.")
    return f"{series_key}|{ref.to_key()}"


def parse_ref_entity_id(entity_id: str) -> tuple[str, RefPeriod]:
    if not entity_id:
        raise ValueError("entity_id is required.")
    parts = entity_id.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid ref entity id: {entity_id}")
    return parts[0], RefPeriod.parse(parts[1])
