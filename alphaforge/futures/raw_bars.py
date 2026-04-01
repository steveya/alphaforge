"""Access local First Rate 5-minute bar directories for non-futures assets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd
from pytz.exceptions import AmbiguousTimeError

from ..data.adapter import SourceAdapterBase
from ..data.context import DataContext
from ..data.query import Query
from ..data.types import FetchResult
from ..store.store import Store
from ..time.calendar import TradingCalendar

_FULL_FILE_SUFFIX = "_full_5min.txt"


@dataclass(frozen=True)
class _BarDatasetSpec:
    config_attr: str
    default_dir_name: str
    layout: str
    entity_col: str = "symbol"
    time_col: str = "available_at_utc"


RAW_BAR_DATASET_SPECS: dict[str, _BarDatasetSpec] = {
    "fx.contract_price_5m": _BarDatasetSpec(
        config_attr="fx_source_dir",
        default_dir_name="fx_contract_price_5m",
        layout="fx_ohlcv",
    ),
    "crypto.contract_price_5m": _BarDatasetSpec(
        config_attr="crypto_source_dir",
        default_dir_name="crypto_contract_price_5m",
        layout="ohlcv",
    ),
    "index.level_5m": _BarDatasetSpec(
        config_attr="index_source_dir",
        default_dir_name="index_level_5m",
        layout="ohlc",
    ),
}


def _maybe_dir(path: Path) -> Path | None:
    return path if path.exists() and path.is_dir() else None


def _localize_bar_starts(
    raw_ts: pd.Series,
    *,
    timezone: str,
) -> pd.Series:
    try:
        return raw_ts.dt.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except AmbiguousTimeError:
        # Sparse files can omit one side of the fallback hour, so pandas cannot infer DST.
        # In that case use the later standard-time interpretation.
        return raw_ts.dt.tz_localize(
            timezone,
            ambiguous=False,
            nonexistent="shift_forward",
        )


@dataclass(frozen=True)
class FirstRateBarsConfig:
    """Resolved configuration for local First Rate raw 5-minute bars."""

    fx_source_dir: Path | None = None
    crypto_source_dir: Path | None = None
    index_source_dir: Path | None = None
    source_timezone: str = "America/New_York"
    bar_minutes: int = 5

    @classmethod
    def from_base_dir(
        cls,
        base_dir: str | Path,
        *,
        source_timezone: str = "America/New_York",
        bar_minutes: int = 5,
    ) -> "FirstRateBarsConfig":
        base_path = Path(base_dir).expanduser().resolve()
        return cls(
            fx_source_dir=_maybe_dir(base_path / "fx_contract_price_5m"),
            crypto_source_dir=_maybe_dir(base_path / "crypto_contract_price_5m"),
            index_source_dir=_maybe_dir(base_path / "index_level_5m"),
            source_timezone=source_timezone,
            bar_minutes=bar_minutes,
        )

    def dataset_roots(self) -> dict[str, Path]:
        roots: dict[str, Path] = {}
        for dataset, spec in RAW_BAR_DATASET_SPECS.items():
            path = getattr(self, spec.config_attr)
            if path is None:
                continue
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Configured First Rate bar directory does not exist for {dataset}: {resolved}"
                )
            if not resolved.is_dir():
                raise ValueError(
                    f"Configured First Rate bar path is not a directory for {dataset}: {resolved}"
                )
            roots[dataset] = resolved

        if not roots:
            raise ValueError(
                "No First Rate raw bar directories configured. "
                "Set at least one of fx_source_dir, crypto_source_dir, or index_source_dir."
            )
        return roots


class FirstRateBarsAdapter(SourceAdapterBase):
    """Read local First Rate 5-minute text files through the SourceAdapter API."""

    source_name = "first_rate_bars"

    def __init__(self, config: FirstRateBarsConfig) -> None:
        self.config = config
        self._dataset_roots = config.dataset_roots()
        self.datasets = frozenset(self._dataset_roots.keys())
        self._entity_files = {
            dataset: self._discover_entity_files(root)
            for dataset, root in self._dataset_roots.items()
        }

    def list_entities(self, dataset: str) -> list[str]:
        if dataset not in self._entity_files:
            raise KeyError(f"Unsupported First Rate bar dataset: {dataset}")
        return sorted(self._entity_files[dataset].keys())

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: object | None = None,
    ) -> FetchResult:
        del max_staleness

        if query.table not in self._dataset_roots:
            raise KeyError(f"Unsupported First Rate bar dataset: {query.table}")

        entity_files = self._entity_files[query.table]
        entities = (
            [str(entity) for entity in query.entities]
            if query.entities is not None
            else sorted(entity_files.keys())
        )

        frames: list[pd.DataFrame] = []
        for entity in entities:
            path = entity_files.get(entity)
            if path is None:
                continue
            frame = self._read_entity_frame(query.table, entity, path)
            if query.start is not None:
                frame = frame[frame["available_at_utc"] >= query.start]
            if query.end is not None:
                frame = frame[frame["available_at_utc"] <= query.end]
            if not frame.empty:
                frames.append(frame)

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["symbol", "available_at_utc"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["symbol", "available_at_utc"])

        df = df.rename(columns={"symbol": "series_key", "available_at_utc": "obs_date"})
        keep = ["series_key", "obs_date"]
        if query.columns:
            for column in query.columns:
                if column in df.columns and column not in keep:
                    keep.append(column)
        else:
            keep.extend(column for column in df.columns if column not in keep)

        return FetchResult(
            data=df[keep].copy(),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
        )

    def _discover_entity_files(self, source_dir: Path) -> dict[str, Path]:
        entity_files: dict[str, Path] = {}
        for path in sorted(source_dir.glob(f"*{_FULL_FILE_SUFFIX}")):
            entity = path.name[: -len(_FULL_FILE_SUFFIX)]
            entity_files[entity] = path
        return entity_files

    @lru_cache(maxsize=512)
    def _read_entity_frame(self, dataset: str, entity: str, path: Path) -> pd.DataFrame:
        spec = RAW_BAR_DATASET_SPECS[dataset]

        if spec.layout == "fx_ohlcv":
            raw = pd.read_csv(
                path,
                header=None,
                names=["trade_date", "trade_time", "open", "high", "low", "close", "volume"],
            )
            raw_ts = pd.to_datetime(
                raw["trade_date"].astype(str).str.strip()
                + " "
                + raw["trade_time"].astype(str).str.strip(),
                format="%Y%m%d %H:%M:%S",
                errors="coerce",
            )
            value_columns = ["open", "high", "low", "close", "volume"]
        elif spec.layout == "ohlcv":
            raw = pd.read_csv(
                path,
                header=None,
                names=["timestamp", "open", "high", "low", "close", "volume"],
            )
            raw_ts = pd.to_datetime(raw["timestamp"], errors="coerce")
            value_columns = ["open", "high", "low", "close", "volume"]
        elif spec.layout == "ohlc":
            raw = pd.read_csv(
                path,
                header=None,
                names=["timestamp", "open", "high", "low", "close"],
            )
            raw_ts = pd.to_datetime(raw["timestamp"], errors="coerce")
            value_columns = ["open", "high", "low", "close"]
        else:
            raise ValueError(f"Unsupported First Rate bar layout: {spec.layout}")

        if raw.empty:
            return pd.DataFrame(columns=["symbol", "available_at_utc"])
        if raw_ts.isna().any():
            raise ValueError(f"Found unparsable timestamps in {path}")

        bar_start_local = _localize_bar_starts(raw_ts, timezone=self.config.source_timezone)
        bar_start_utc = bar_start_local.dt.tz_convert("UTC")
        frame = pd.DataFrame(
            {
                "symbol": entity,
                "source_file": path.name,
                "bar_start_utc": bar_start_utc,
            }
        )
        frame["bar_end_utc"] = frame["bar_start_utc"] + pd.Timedelta(minutes=self.config.bar_minutes)
        frame["available_at_utc"] = frame["bar_end_utc"]

        for column in value_columns:
            frame[column] = pd.to_numeric(raw[column], errors="raise")

        return frame.sort_values("available_at_utc").reset_index(drop=True)


def build_first_rate_bars_context(
    config: FirstRateBarsConfig,
    *,
    calendars: Mapping[str, TradingCalendar] | None = None,
    store: Store | None = None,
    source_name: str = "first_rate_bars",
) -> DataContext:
    """Build a DataContext for locally mounted First Rate 5-minute bar directories."""

    adapter = FirstRateBarsAdapter(config)
    return DataContext(
        sources={},
        calendars=dict(calendars or {}),
        store=store,
        adapters={source_name: adapter},
        default_sources={dataset: source_name for dataset in adapter.datasets},
    )
