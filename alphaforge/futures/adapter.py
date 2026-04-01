"""SourceAdapter for persisted First Rate Data futures artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.adapter import SourceAdapterBase
from ..data.query import Query
from ..data.types import FetchResult

DATASET_SPECS = {
    "futures.contract_5m_sparse": {
        "dir": "contract_5m_sparse",
        "entity_col": "contract_id",
        "time_col": "available_at_utc",
    },
    "futures.contract_5m_dense": {
        "dir": "contract_5m_dense",
        "entity_col": "contract_id",
        "time_col": "available_at_utc",
    },
    "futures.contract_eod": {
        "dir": "contract_eod",
        "entity_col": "contract_id",
        "time_col": "session_close_utc",
    },
    "futures.continuous_5m_execution": {
        "dir": "continuous_5m_execution",
        "entity_col": "root_symbol",
        "time_col": "available_at_utc",
    },
    "futures.continuous_eod_research": {
        "dir": "continuous_eod_research",
        "entity_col": "root_symbol",
        "time_col": "session_close_utc",
    },
}


class FirstRateFuturesAdapter(SourceAdapterBase):
    """Read persisted local futures artifacts via the SourceAdapter protocol."""

    source_name = "first_rate_futures"
    datasets = frozenset(DATASET_SPECS.keys())

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self._manifest_path = self.artifact_root / "manifests" / "entity_manifest.parquet"

    def list_entities(self, dataset: str) -> list[str]:
        if not self._manifest_path.exists():
            return []
        manifest = pd.read_parquet(self._manifest_path)
        subset = manifest[manifest["dataset"] == dataset]
        if subset.empty:
            return []
        return subset["entity_id"].astype(str).sort_values().tolist()

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[object] = None,
    ) -> FetchResult:
        del max_staleness
        spec = DATASET_SPECS.get(query.table)
        if spec is None:
            raise KeyError(f"Unsupported futures dataset: {query.table}")

        data_path = self.artifact_root / spec["dir"] / "data.parquet"
        if not data_path.exists():
            return FetchResult(
                data=pd.DataFrame(columns=["series_key", "obs_date"]),
                source=self.source_name,
                dataset=query.table,
                is_pit=False,
                cached_at=None,
            )

        df = pd.read_parquet(data_path)
        entity_col = str(spec["entity_col"])
        time_col = str(spec["time_col"])

        if query.entities:
            df = df[df[entity_col].isin(list(query.entities))]
        if query.start is not None:
            df = df[df[time_col] >= query.start]
        if query.end is not None:
            df = df[df[time_col] <= query.end]

        df = df.sort_values([entity_col, time_col]).reset_index(drop=True)
        df = df.rename(columns={entity_col: "series_key", time_col: "obs_date"})

        keep = ["series_key", "obs_date"]
        if query.columns:
            for column in query.columns:
                if column in df.columns and column not in keep:
                    keep.append(column)
        else:
            keep.extend(
                column for column in df.columns if column not in keep
            )

        return FetchResult(
            data=df[keep].copy(),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
        )
