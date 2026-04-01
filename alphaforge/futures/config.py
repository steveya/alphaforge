"""Configuration for local First Rate Data futures ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_CONFIG = "ALPHAFORGE_FRD_FUTURES_CONFIG"
_ENV_SOURCE_DIR = "ALPHAFORGE_FRD_FUTURES_SOURCE_DIR"
_ENV_ARTIFACT_ROOT = "ALPHAFORGE_FRD_FUTURES_ARTIFACT_ROOT"
_ENV_METADATA_PATH = "ALPHAFORGE_FRD_FUTURES_METADATA_PATH"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in config file: {path}")
    return raw


def _coerce_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


@dataclass(frozen=True)
class FirstRateFuturesConfig:
    """Resolved futures loader configuration.

    Resolution order:
    1. explicit arguments
    2. YAML config file entries
    3. environment variables
    """

    source_dir: Path
    artifact_root: Path
    metadata_path: Path | None = None
    config_path: Path | None = None
    source_timezone: str = "America/New_York"
    bar_minutes: int = 5
    roll_min_consecutive_sessions: int = 2

    @classmethod
    def from_yaml(
        cls,
        config_path: str | os.PathLike[str],
        *,
        source_dir: str | os.PathLike[str] | None = None,
        artifact_root: str | os.PathLike[str] | None = None,
        metadata_path: str | os.PathLike[str] | None = None,
        source_timezone: str | None = None,
        bar_minutes: int | None = None,
        roll_min_consecutive_sessions: int | None = None,
    ) -> "FirstRateFuturesConfig":
        return cls.resolve(
            config_path=config_path,
            source_dir=source_dir,
            artifact_root=artifact_root,
            metadata_path=metadata_path,
            source_timezone=source_timezone,
            bar_minutes=bar_minutes,
            roll_min_consecutive_sessions=roll_min_consecutive_sessions,
        )

    @classmethod
    def from_env(
        cls,
        *,
        config_path: str | os.PathLike[str] | None = None,
        source_dir: str | os.PathLike[str] | None = None,
        artifact_root: str | os.PathLike[str] | None = None,
        metadata_path: str | os.PathLike[str] | None = None,
        source_timezone: str | None = None,
        bar_minutes: int | None = None,
        roll_min_consecutive_sessions: int | None = None,
    ) -> "FirstRateFuturesConfig":
        return cls.resolve(
            config_path=config_path or os.environ.get(_ENV_CONFIG),
            source_dir=source_dir,
            artifact_root=artifact_root,
            metadata_path=metadata_path,
            source_timezone=source_timezone,
            bar_minutes=bar_minutes,
            roll_min_consecutive_sessions=roll_min_consecutive_sessions,
        )

    @classmethod
    def resolve(
        cls,
        *,
        config_path: str | os.PathLike[str] | None = None,
        source_dir: str | os.PathLike[str] | None = None,
        artifact_root: str | os.PathLike[str] | None = None,
        metadata_path: str | os.PathLike[str] | None = None,
        source_timezone: str | None = None,
        bar_minutes: int | None = None,
        roll_min_consecutive_sessions: int | None = None,
    ) -> "FirstRateFuturesConfig":
        yaml_path = _coerce_path(config_path)
        yaml_cfg = _read_yaml(yaml_path) if yaml_path is not None else {}

        resolved_source_dir = _coerce_path(
            source_dir
            or yaml_cfg.get("source_dir")
            or os.environ.get(_ENV_SOURCE_DIR)
        )
        resolved_artifact_root = _coerce_path(
            artifact_root
            or yaml_cfg.get("artifact_root")
            or os.environ.get(_ENV_ARTIFACT_ROOT)
        )
        resolved_metadata_path = _coerce_path(
            metadata_path
            or yaml_cfg.get("metadata_path")
            or os.environ.get(_ENV_METADATA_PATH)
        )
        resolved_source_tz = (
            source_timezone
            or yaml_cfg.get("source_timezone")
            or "America/New_York"
        )
        resolved_bar_minutes = int(
            bar_minutes or yaml_cfg.get("bar_minutes") or 5
        )
        resolved_roll_sessions = int(
            roll_min_consecutive_sessions
            or yaml_cfg.get("roll_min_consecutive_sessions")
            or 2
        )

        missing: list[str] = []
        if resolved_source_dir is None:
            missing.append(_ENV_SOURCE_DIR)
        if resolved_artifact_root is None:
            missing.append(_ENV_ARTIFACT_ROOT)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Missing required futures configuration values. "
                f"Set them explicitly, via YAML, or via env vars: {joined}"
            )

        return cls(
            source_dir=resolved_source_dir.resolve(),
            artifact_root=resolved_artifact_root.resolve(),
            metadata_path=resolved_metadata_path.resolve()
            if resolved_metadata_path is not None
            else None,
            config_path=yaml_path.resolve() if yaml_path is not None else None,
            source_timezone=resolved_source_tz,
            bar_minutes=resolved_bar_minutes,
            roll_min_consecutive_sessions=resolved_roll_sessions,
        )

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "source_dir": str(self.source_dir),
            "artifact_root": str(self.artifact_root),
            "metadata_path": None if self.metadata_path is None else str(self.metadata_path),
            "config_path": None if self.config_path is None else str(self.config_path),
            "source_timezone": self.source_timezone,
            "bar_minutes": self.bar_minutes,
            "roll_min_consecutive_sessions": self.roll_min_consecutive_sessions,
        }
