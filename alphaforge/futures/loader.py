"""Local First Rate Data futures ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .adapter import DATASET_SPECS
from .config import FirstRateFuturesConfig
from .metadata import FirstRateFuturesRootMetadata, resolve_root_metadata

_CONTRACT_RE = re.compile(
    r"^(?P<root>[A-Z0-9]+)_(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{2})_5min\.txt$"
)
_MONTH_TO_NUM = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


@dataclass(frozen=True)
class FirstRateFuturesArtifacts:
    artifact_root: Path
    manifest_path: Path
    roll_schedule_path: Path
    dataset_paths: dict[str, Path]


@dataclass(frozen=True)
class _ContractFile:
    path: Path
    root_symbol: str
    month_code: str
    contract_year: int
    contract_month: int

    @property
    def contract_id(self) -> str:
        return f"{self.root_symbol}_{self.month_code}{str(self.contract_year)[-2:]}"

    @property
    def contract_sort_key(self) -> tuple[int, int]:
        return self.contract_year, self.contract_month


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def _coerce_session_date(
    local_ts: pd.Timestamp,
    metadata: FirstRateFuturesRootMetadata,
) -> pd.Timestamp:
    open_hour, open_minute = _parse_hhmm(metadata.session_open_local)
    local_clock = (local_ts.hour, local_ts.minute)
    date_value = local_ts.date()
    if local_clock >= (open_hour, open_minute):
        date_value = date_value + pd.Timedelta(days=1)
    return pd.Timestamp(date_value).normalize()


def _session_bar_starts_utc(
    session_date: pd.Timestamp,
    metadata: FirstRateFuturesRootMetadata,
    *,
    bar_minutes: int,
) -> pd.DatetimeIndex:
    open_hour, open_minute = _parse_hhmm(metadata.session_open_local)
    close_hour, close_minute = _parse_hhmm(metadata.session_close_local)

    close_local = pd.Timestamp(session_date).tz_localize(metadata.timezone) + pd.Timedelta(
        hours=close_hour,
        minutes=close_minute,
    )
    open_local = (close_local - pd.Timedelta(days=1)).normalize() + pd.Timedelta(
        hours=open_hour,
        minutes=open_minute,
    )
    last_start_local = close_local - pd.Timedelta(minutes=bar_minutes)
    return pd.date_range(
        start=open_local,
        end=last_start_local,
        freq=f"{bar_minutes}min",
    ).tz_convert("UTC")


def _session_close_utc(
    session_date: pd.Timestamp,
    metadata: FirstRateFuturesRootMetadata,
) -> pd.Timestamp:
    close_hour, close_minute = _parse_hhmm(metadata.session_close_local)
    close_local = pd.Timestamp(session_date).tz_localize(metadata.timezone) + pd.Timedelta(
        hours=close_hour,
        minutes=close_minute,
    )
    return close_local.tz_convert("UTC")


class FirstRateFuturesLoader:
    """Ingest a local directory of First Rate Data contract files."""

    def __init__(self, config: FirstRateFuturesConfig) -> None:
        self.config = config

    @classmethod
    def from_config(
        cls,
        config: FirstRateFuturesConfig | str | Path,
    ) -> "FirstRateFuturesLoader":
        if isinstance(config, FirstRateFuturesConfig):
            return cls(config)
        return cls(FirstRateFuturesConfig.from_yaml(config))

    @classmethod
    def from_env(cls) -> "FirstRateFuturesLoader":
        return cls(FirstRateFuturesConfig.from_env())

    def ingest(self) -> FirstRateFuturesArtifacts:
        contract_files = self._discover_contract_files()
        sparse_frames: list[pd.DataFrame] = []
        dense_frames: list[pd.DataFrame] = []
        eod_frames: list[pd.DataFrame] = []
        contract_catalog_rows: list[dict[str, object]] = []

        for contract in contract_files:
            metadata = resolve_root_metadata(
                contract.root_symbol,
                metadata_path=self.config.metadata_path,
            )
            sparse = self._read_contract_file(contract, metadata)
            dense = self._build_dense_frame(sparse, metadata)
            eod = self._build_eod_frame(dense, sparse, metadata)

            sparse_frames.append(sparse)
            dense_frames.append(dense)
            eod_frames.append(eod)

            contract_catalog_rows.append(
                {
                    "root_symbol": contract.root_symbol,
                    "contract_id": contract.contract_id,
                    "contract_year": contract.contract_year,
                    "contract_month": contract.contract_month,
                    "first_session_date": eod["session_date"].min() if not eod.empty else pd.NaT,
                    "last_session_date": eod["session_date"].max() if not eod.empty else pd.NaT,
                }
            )

        contract_sparse = pd.concat(sparse_frames, ignore_index=True) if sparse_frames else pd.DataFrame()
        contract_dense = pd.concat(dense_frames, ignore_index=True) if dense_frames else pd.DataFrame()
        contract_eod = pd.concat(eod_frames, ignore_index=True) if eod_frames else pd.DataFrame()
        contract_catalog = pd.DataFrame(contract_catalog_rows)

        roll_schedule = self._build_roll_schedule(contract_catalog, contract_eod)
        continuous_exec = self._build_continuous_execution(contract_dense, roll_schedule)
        continuous_eod = self._build_continuous_eod(contract_eod, roll_schedule)

        dataset_frames = {
            "futures.contract_5m_sparse": contract_sparse,
            "futures.contract_5m_dense": contract_dense,
            "futures.contract_eod": contract_eod,
            "futures.continuous_5m_execution": continuous_exec,
            "futures.continuous_eod_research": continuous_eod,
        }

        dataset_paths = self._write_outputs(dataset_frames, roll_schedule)
        manifest_path = self._write_manifest(dataset_frames, dataset_paths)

        return FirstRateFuturesArtifacts(
            artifact_root=self.config.artifact_root,
            manifest_path=manifest_path,
            roll_schedule_path=dataset_paths["roll_schedule"],
            dataset_paths=dataset_paths,
        )

    def _discover_contract_files(self) -> list[_ContractFile]:
        source_dir = self.config.source_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"First Rate futures source_dir does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise ValueError(f"First Rate futures source_dir is not a directory: {source_dir}")

        nested_dirs = [path for path in source_dir.iterdir() if path.is_dir()]
        if nested_dirs:
            names = ", ".join(path.name for path in nested_dirs[:5])
            raise ValueError(
                "Expected a flat source_dir containing only contract files. "
                f"Found nested directories: {names}"
            )

        files = sorted(path for path in source_dir.iterdir() if path.is_file())
        if not files:
            raise ValueError(f"No contract files found in source_dir: {source_dir}")

        contracts: list[_ContractFile] = []
        invalid: list[str] = []
        for path in files:
            match = _CONTRACT_RE.match(path.name)
            if match is None:
                invalid.append(path.name)
                continue
            year = 2000 + int(match.group("year"))
            month_code = match.group("month")
            contracts.append(
                _ContractFile(
                    path=path,
                    root_symbol=match.group("root"),
                    month_code=month_code,
                    contract_year=year,
                    contract_month=_MONTH_TO_NUM[month_code],
                )
            )

        if invalid:
            preview = ", ".join(sorted(invalid)[:5])
            raise ValueError(
                "Invalid First Rate contract filenames. "
                "Expected <ROOT>_<MONTH><YY>_5min.txt. "
                f"Examples of invalid files: {preview}"
            )

        unsupported = sorted(
            {
                contract.root_symbol
                for contract in contracts
                if self._is_unsupported_root(contract.root_symbol)
            }
        )
        if unsupported:
            preview = ", ".join(unsupported[:10])
            raise ValueError(
                "Unsupported futures roots found in source_dir. "
                f"Provide a metadata override YAML for: {preview}"
            )

        return sorted(contracts, key=lambda item: (item.root_symbol, item.contract_sort_key))

    def _is_unsupported_root(self, root_symbol: str) -> bool:
        try:
            resolve_root_metadata(root_symbol, metadata_path=self.config.metadata_path)
        except KeyError:
            return True
        return False

    def _read_contract_file(
        self,
        contract: _ContractFile,
        metadata: FirstRateFuturesRootMetadata,
    ) -> pd.DataFrame:
        df = pd.read_csv(
            contract.path,
            header=None,
            names=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if df.shape[1] != 6:
            raise ValueError(f"Expected 6 columns in {contract.path.name}, got {df.shape[1]}")
        if df.empty:
            raise ValueError(f"Contract file is empty: {contract.path.name}")

        raw_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if raw_ts.isna().any():
            raise ValueError(f"Found unparsable timestamps in {contract.path.name}")
        bar_start_local = raw_ts.dt.tz_localize(
            metadata.timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        bar_start_utc = bar_start_local.dt.tz_convert("UTC")
        if bar_start_utc.duplicated().any():
            raise ValueError(f"Duplicate timestamps found in {contract.path.name}")

        frame = pd.DataFrame(
            {
                "root_symbol": contract.root_symbol,
                "contract_id": contract.contract_id,
                "venue": metadata.venue,
                "source_file": contract.path.name,
                "bar_start_utc": bar_start_utc,
                "open": pd.to_numeric(df["open"], errors="raise") * metadata.price_scale,
                "high": pd.to_numeric(df["high"], errors="raise") * metadata.price_scale,
                "low": pd.to_numeric(df["low"], errors="raise") * metadata.price_scale,
                "close": pd.to_numeric(df["close"], errors="raise") * metadata.price_scale,
                "volume": pd.to_numeric(df["volume"], errors="raise").astype(float),
            }
        ).sort_values("bar_start_utc")
        frame["bar_end_utc"] = frame["bar_start_utc"] + pd.Timedelta(minutes=self.config.bar_minutes)
        frame["available_at_utc"] = frame["bar_end_utc"]
        local_ts = frame["bar_start_utc"].dt.tz_convert(metadata.timezone)
        frame["session_date"] = local_ts.apply(lambda ts: _coerce_session_date(ts, metadata))
        frame["session_close_utc"] = frame["session_date"].apply(
            lambda date_value: _session_close_utc(date_value, metadata)
        )
        return frame.reset_index(drop=True)

    def _build_dense_frame(
        self,
        sparse: pd.DataFrame,
        metadata: FirstRateFuturesRootMetadata,
    ) -> pd.DataFrame:
        if sparse.empty:
            return sparse.copy()

        session_dates = sorted(pd.Timestamp(value) for value in sparse["session_date"].drop_duplicates())
        grids = [
            _session_bar_starts_utc(
                session_date,
                metadata,
                bar_minutes=self.config.bar_minutes,
            )
            for session_date in session_dates
        ]
        if grids:
            full_index = pd.DatetimeIndex(sorted({ts for grid in grids for ts in grid}))
        else:
            full_index = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")

        first_obs = sparse["bar_start_utc"].min()
        last_obs = sparse["bar_start_utc"].max()
        full_index = full_index[(full_index >= first_obs) & (full_index <= last_obs)]

        dense = sparse.set_index("bar_start_utc").reindex(full_index)
        observed_mask = dense["close"].notna()
        prev_close = dense["close"].ffill()

        for column in ("open", "high", "low", "close"):
            dense[column] = dense[column].where(observed_mask, prev_close)
        dense["volume"] = dense["volume"].fillna(0.0)
        dense["is_synthetic_bar"] = ~observed_mask
        dense["root_symbol"] = dense["root_symbol"].fillna(sparse["root_symbol"].iloc[0])
        dense["contract_id"] = dense["contract_id"].fillna(sparse["contract_id"].iloc[0])
        dense["venue"] = dense["venue"].fillna(sparse["venue"].iloc[0])
        dense["source_file"] = dense["source_file"].fillna(sparse["source_file"].iloc[0])

        dense = dense.reset_index().rename(columns={"index": "bar_start_utc"})
        dense["bar_end_utc"] = dense["bar_start_utc"] + pd.Timedelta(minutes=self.config.bar_minutes)
        dense["available_at_utc"] = dense["bar_end_utc"]
        local_ts = dense["bar_start_utc"].dt.tz_convert(metadata.timezone)
        dense["session_date"] = local_ts.apply(lambda ts: _coerce_session_date(ts, metadata))
        dense["session_close_utc"] = dense["session_date"].apply(
            lambda date_value: _session_close_utc(date_value, metadata)
        )
        return dense.reset_index(drop=True)

    def _build_eod_frame(
        self,
        dense: pd.DataFrame,
        sparse: pd.DataFrame,
        metadata: FirstRateFuturesRootMetadata,
    ) -> pd.DataFrame:
        if dense.empty:
            return pd.DataFrame()

        observed_counts = sparse.groupby("session_date").size().to_dict()
        rows: list[dict[str, object]] = []
        for session_date, group in dense.groupby("session_date", sort=True):
            group = group.sort_values("bar_start_utc")
            expected_bars = len(
                _session_bar_starts_utc(
                    pd.Timestamp(session_date),
                    metadata,
                    bar_minutes=self.config.bar_minutes,
                )
            )
            rows.append(
                {
                    "root_symbol": group["root_symbol"].iloc[0],
                    "contract_id": group["contract_id"].iloc[0],
                    "venue": group["venue"].iloc[0],
                    "session_date": pd.Timestamp(session_date),
                    "session_open_utc": group["bar_start_utc"].iloc[0],
                    "session_close_utc": _session_close_utc(pd.Timestamp(session_date), metadata),
                    "open": float(group["open"].iloc[0]),
                    "high": float(group["high"].max()),
                    "low": float(group["low"].min()),
                    "close": float(group["close"].iloc[-1]),
                    "volume": float(group["volume"].sum()),
                    "bar_count": int(len(group)),
                    "observed_bar_count": int(observed_counts.get(session_date, 0)),
                    "expected_bar_count": int(expected_bars),
                    "is_partial_session": bool(len(group) < expected_bars),
                }
            )
        return pd.DataFrame(rows).sort_values(["contract_id", "session_date"]).reset_index(drop=True)

    def _build_roll_schedule(
        self,
        contract_catalog: pd.DataFrame,
        contract_eod: pd.DataFrame,
    ) -> pd.DataFrame:
        if contract_catalog.empty or contract_eod.empty:
            return pd.DataFrame(
                columns=[
                    "root_symbol",
                    "active_contract_id",
                    "start_session_date",
                    "end_session_date",
                    "rolled_from_contract_id",
                    "roll_trigger_session_date",
                    "roll_reason",
                ]
            )

        rows: list[dict[str, object]] = []
        for root_symbol, contracts in contract_catalog.groupby("root_symbol", sort=True):
            contracts = contracts.sort_values(["contract_year", "contract_month"]).reset_index(drop=True)
            current_contract_id = str(contracts.iloc[0]["contract_id"])
            current_start = pd.Timestamp(contracts.iloc[0]["first_session_date"])
            rolled_from_contract_id: str | None = None

            for i in range(len(contracts) - 1):
                current_id = str(contracts.iloc[i]["contract_id"])
                next_id = str(contracts.iloc[i + 1]["contract_id"])
                current_eod = contract_eod[contract_eod["contract_id"] == current_id].sort_values("session_date")
                next_eod = contract_eod[contract_eod["contract_id"] == next_id].sort_values("session_date")
                if current_eod.empty or next_eod.empty:
                    continue

                trigger_date, next_start, reason = self._find_roll_transition(current_eod, next_eod)
                current_end = current_eod[current_eod["session_date"] < next_start]["session_date"].max()
                rows.append(
                    {
                        "root_symbol": root_symbol,
                        "active_contract_id": current_id,
                        "start_session_date": current_start,
                        "end_session_date": current_end,
                        "rolled_from_contract_id": rolled_from_contract_id,
                        "roll_trigger_session_date": trigger_date,
                        "roll_reason": reason,
                    }
                )
                rolled_from_contract_id = current_id
                current_contract_id = next_id
                current_start = next_start

            final_eod = contract_eod[contract_eod["contract_id"] == current_contract_id].sort_values("session_date")
            if not final_eod.empty:
                rows.append(
                    {
                        "root_symbol": root_symbol,
                        "active_contract_id": current_contract_id,
                        "start_session_date": current_start,
                        "end_session_date": final_eod["session_date"].max(),
                        "rolled_from_contract_id": rolled_from_contract_id,
                        "roll_trigger_session_date": pd.NaT,
                        "roll_reason": "active_final",
                    }
                )

        schedule = pd.DataFrame(rows)
        if schedule.empty:
            return schedule
        return schedule.sort_values(["root_symbol", "start_session_date"]).reset_index(drop=True)

    def _find_roll_transition(
        self,
        current_eod: pd.DataFrame,
        next_eod: pd.DataFrame,
    ) -> tuple[pd.Timestamp, pd.Timestamp, str]:
        overlap = current_eod[["session_date", "volume"]].merge(
            next_eod[["session_date", "volume"]],
            on="session_date",
            how="inner",
            suffixes=("_current", "_next"),
        ).sort_values("session_date")

        trigger_date: pd.Timestamp | None = None
        streak = 0
        for row in overlap.itertuples(index=False):
            if float(row.volume_next) > float(row.volume_current):
                streak += 1
            else:
                streak = 0
            if streak >= self.config.roll_min_consecutive_sessions:
                trigger_date = pd.Timestamp(row.session_date)
                break

        if trigger_date is None:
            trigger_date = pd.Timestamp(current_eod["session_date"].max())
            next_dates = next_eod[next_eod["session_date"] > trigger_date]["session_date"]
            next_start = (
                pd.Timestamp(next_dates.min())
                if not next_dates.empty
                else pd.Timestamp(next_eod["session_date"].min())
            )
            return trigger_date, next_start, "fallback_last_session"

        next_dates = next_eod[next_eod["session_date"] > trigger_date]["session_date"]
        next_start = (
            pd.Timestamp(next_dates.min())
            if not next_dates.empty
            else pd.Timestamp(next_eod["session_date"].min())
        )
        return trigger_date, next_start, "volume_crossover"

    def _build_continuous_execution(
        self,
        contract_dense: pd.DataFrame,
        roll_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        if contract_dense.empty or roll_schedule.empty:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for row in roll_schedule.itertuples(index=False):
            subset = contract_dense[
                (contract_dense["contract_id"] == row.active_contract_id)
                & (contract_dense["session_date"] >= row.start_session_date)
                & (contract_dense["session_date"] <= row.end_session_date)
            ].copy()
            if subset.empty:
                continue
            subset["active_contract_id"] = row.active_contract_id
            subset["roll_flag"] = False
            if pd.notna(row.rolled_from_contract_id):
                first_idx = subset.index.min()
                subset.loc[first_idx, "roll_flag"] = True
            frames.append(subset)
        if not frames:
            return pd.DataFrame()
        continuous = pd.concat(frames, ignore_index=True)
        return continuous.sort_values(["root_symbol", "available_at_utc"]).reset_index(drop=True)

    def _build_continuous_eod(
        self,
        contract_eod: pd.DataFrame,
        roll_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        if contract_eod.empty or roll_schedule.empty:
            return pd.DataFrame()

        selected_frames: list[pd.DataFrame] = []
        roll_events: list[dict[str, object]] = []

        for root_symbol, schedule in roll_schedule.groupby("root_symbol", sort=True):
            schedule = schedule.sort_values("start_session_date").reset_index(drop=True)
            prior_end_row: pd.Series | None = None
            for _, row in schedule.iterrows():
                subset = contract_eod[
                    (contract_eod["contract_id"] == row["active_contract_id"])
                    & (contract_eod["session_date"] >= row["start_session_date"])
                    & (contract_eod["session_date"] <= row["end_session_date"])
                ].copy()
                if subset.empty:
                    continue
                subset["active_contract_id"] = row["active_contract_id"]
                subset["roll_flag"] = False
                first_subset_row = subset.index.min()
                if pd.notna(row["rolled_from_contract_id"]):
                    subset.loc[first_subset_row, "roll_flag"] = True
                    new_open = float(subset.iloc[0]["open"])
                    prior_close = 1.0 if prior_end_row is None else float(prior_end_row["close"])
                    factor = 1.0 if prior_close == 0 else new_open / prior_close
                    roll_events.append(
                        {
                            "root_symbol": root_symbol,
                            "effective_session_date": pd.Timestamp(row["start_session_date"]),
                            "adjustment_factor": factor,
                        }
                    )
                prior_end_row = subset.sort_values("session_date").iloc[-1]
                selected_frames.append(subset)

        if not selected_frames:
            return pd.DataFrame()

        continuous = pd.concat(selected_frames, ignore_index=True)
        continuous = continuous.sort_values(["root_symbol", "session_date"]).reset_index(drop=True)
        continuous["adjustment_factor"] = 1.0
        continuous["cumulative_adjustment_factor"] = 1.0

        for event in roll_events:
            mask = (
                (continuous["root_symbol"] == event["root_symbol"])
                & (continuous["session_date"] < event["effective_session_date"])
            )
            continuous.loc[mask, "cumulative_adjustment_factor"] *= float(event["adjustment_factor"])
            start_mask = (
                (continuous["root_symbol"] == event["root_symbol"])
                & (continuous["session_date"] == event["effective_session_date"])
            )
            continuous.loc[start_mask, "adjustment_factor"] = float(event["adjustment_factor"])

        for column in ("open", "high", "low", "close"):
            continuous[f"raw_{column}"] = continuous[column]
            continuous[column] = continuous[column] * continuous["cumulative_adjustment_factor"]

        return continuous.reset_index(drop=True)

    def _write_outputs(
        self,
        dataset_frames: dict[str, pd.DataFrame],
        roll_schedule: pd.DataFrame,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.config.artifact_root / "manifests").mkdir(parents=True, exist_ok=True)

        for dataset, spec in DATASET_SPECS.items():
            out_dir = self.config.artifact_root / spec["dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "data.parquet"
            dataset_frames[dataset].to_parquet(out_path, index=False)
            paths[dataset] = out_path

        roll_dir = self.config.artifact_root / "roll_schedules"
        roll_dir.mkdir(parents=True, exist_ok=True)
        roll_path = roll_dir / "data.parquet"
        roll_schedule.to_parquet(roll_path, index=False)
        paths["roll_schedule"] = roll_path
        return paths

    def _write_manifest(
        self,
        dataset_frames: dict[str, pd.DataFrame],
        dataset_paths: dict[str, Path],
    ) -> Path:
        manifest_rows: list[dict[str, object]] = []
        for dataset, frame in dataset_frames.items():
            spec = DATASET_SPECS[dataset]
            entity_col = str(spec["entity_col"])
            time_col = str(spec["time_col"])
            if frame.empty:
                continue
            grouped = frame.groupby(entity_col).agg(
                row_count=(entity_col, "size"),
                min_obs_date=(time_col, "min"),
                max_obs_date=(time_col, "max"),
            )
            for entity_id, row in grouped.reset_index().iterrows():
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "entity_id": row[entity_col],
                        "row_count": int(row["row_count"]),
                        "min_obs_date": row["min_obs_date"],
                        "max_obs_date": row["max_obs_date"],
                        "path": str(dataset_paths[dataset]),
                    }
                )

        manifest = pd.DataFrame(manifest_rows)
        manifest_path = self.config.artifact_root / "manifests" / "entity_manifest.parquet"
        manifest.to_parquet(manifest_path, index=False)
        return manifest_path
