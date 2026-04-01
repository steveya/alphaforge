"""Metadata loading for local First Rate Data futures roots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FirstRateFuturesRootMetadata:
    root_symbol: str
    venue: str
    timezone: str
    session_open_local: str
    session_close_local: str
    price_scale: float = 1.0


@dataclass(frozen=True)
class _PatternMetadata:
    regex: re.Pattern[str]
    venue: str
    timezone: str
    session_open_local: str
    session_close_local: str
    price_scale: float


def _default_metadata_text() -> str:
    resource = resources.files("alphaforge.futures").joinpath(
        "resources/us_root_metadata.yaml"
    )
    return resource.read_text(encoding="utf-8")


def _load_yaml_mapping(metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None:
        raw = yaml.safe_load(_default_metadata_text())
    else:
        raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Futures metadata YAML must be a mapping.")
    return raw


def _load_metadata_bundle(
    metadata_path: Path | None = None,
) -> tuple[dict[str, FirstRateFuturesRootMetadata], list[_PatternMetadata]]:
    raw = _load_yaml_mapping(metadata_path)
    defaults = dict(raw.get("defaults", {}))
    roots_raw = raw.get("roots", {}) or {}
    patterns_raw = raw.get("patterns", []) or []

    roots: dict[str, FirstRateFuturesRootMetadata] = {}
    for root_symbol, payload in roots_raw.items():
        cfg = dict(defaults)
        cfg.update(payload or {})
        roots[root_symbol] = FirstRateFuturesRootMetadata(
            root_symbol=root_symbol,
            venue=str(cfg["venue"]),
            timezone=str(cfg["timezone"]),
            session_open_local=str(cfg["session_open_local"]),
            session_close_local=str(cfg["session_close_local"]),
            price_scale=float(cfg.get("price_scale", 1.0)),
        )

    patterns: list[_PatternMetadata] = []
    for payload in patterns_raw:
        cfg = dict(defaults)
        cfg.update(payload or {})
        regex = str(cfg.get("regex", "")).strip()
        if not regex:
            continue
        patterns.append(
            _PatternMetadata(
                regex=re.compile(regex),
                venue=str(cfg["venue"]),
                timezone=str(cfg["timezone"]),
                session_open_local=str(cfg["session_open_local"]),
                session_close_local=str(cfg["session_close_local"]),
                price_scale=float(cfg.get("price_scale", 1.0)),
            )
        )

    return roots, patterns


def load_first_rate_futures_metadata(
    metadata_path: Path | None = None,
) -> dict[str, FirstRateFuturesRootMetadata]:
    roots, _ = _load_metadata_bundle(metadata_path)
    return roots


def resolve_root_metadata(
    root_symbol: str,
    *,
    metadata_path: Path | None = None,
) -> FirstRateFuturesRootMetadata:
    roots, patterns = _load_metadata_bundle(metadata_path)
    if root_symbol in roots:
        return roots[root_symbol]

    for pattern in patterns:
        if pattern.regex.match(root_symbol):
            return FirstRateFuturesRootMetadata(
                root_symbol=root_symbol,
                venue=pattern.venue,
                timezone=pattern.timezone,
                session_open_local=pattern.session_open_local,
                session_close_local=pattern.session_close_local,
                price_scale=pattern.price_scale,
            )

    raise KeyError(
        f"Unsupported futures root '{root_symbol}'. "
        "Add it to the package metadata or provide a metadata override YAML."
    )
