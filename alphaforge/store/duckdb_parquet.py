from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from alphaforge.features.frame import FeatureFrame, Artifact
from alphaforge.features.realization import FitState
from alphaforge.pit.accessor import ensure_pit_table


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class DuckDBParquetStore:
    """
    Store FeatureFrames as Parquet on disk, indexed by DuckDB.

    Layout:
      root/
        alphaforge.duckdb
        frames/<realization_id>/X.parquet
        frames/<realization_id>/catalog.parquet
        frames/<realization_id>/meta.json
        states/<state_id>/payload.bin
        states/<state_id>/meta.json
    """

    root: str
    duckdb_path: Optional[str] = None
    _cached_conn: Optional[duckdb.DuckDBPyConnection] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        self.root_path = Path(self.root).resolve()
        _ensure_dir(self.root_path)

        if self.duckdb_path is None:
            self.duckdb_path = str(self.root_path / "alphaforge.duckdb")

        self._init_db()

    # ---------------------------
    # DuckDB initialization
    # ---------------------------
    def _conn(self):
        return duckdb.connect(self.duckdb_path)

    def conn(self):
        if self._cached_conn is None:
            self._cached_conn = self._conn()
        return self._cached_conn

    def _init_db(self):
        con = self.conn()
        con.execute(
            """
        CREATE TABLE IF NOT EXISTS frames (
            realization_id VARCHAR PRIMARY KEY,
            x_path VARCHAR NOT NULL,
            catalog_path VARCHAR NOT NULL,
            meta_path VARCHAR NOT NULL,
            created_utc VARCHAR NOT NULL,
            n_rows BIGINT,
            n_cols BIGINT
        );
        """
        )
        con.execute(
            """
        CREATE TABLE IF NOT EXISTS states (
            state_id VARCHAR PRIMARY KEY,
            payload_path VARCHAR NOT NULL,
            meta_path VARCHAR NOT NULL,
            created_utc VARCHAR NOT NULL
        );
        """
        )
        ensure_pit_table(con)

    # ---------------------------
    # Frames
    # ---------------------------
    def _frame_dir(self, realization_id: str) -> Path:
        return self.root_path / "frames" / realization_id

    def exists_frame(self, realization_id: str) -> bool:
        con = self.conn()
        row = con.execute(
            "SELECT 1 FROM frames WHERE realization_id = ? LIMIT 1",
            [realization_id],
        ).fetchone()
        return row is not None

    def get_frame(self, realization_id: str) -> Optional[FeatureFrame]:
        con = self.conn()
        row = con.execute(
            "SELECT x_path, catalog_path, meta_path FROM frames WHERE realization_id = ?",
            [realization_id],
        ).fetchone()

        if row is None:
            return None

        x_path, catalog_path, meta_path = row
        X = pd.read_parquet(x_path)
        catalog = pd.read_parquet(catalog_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Ensure catalog index is feature_id if stored as a column
        if "feature_id" in catalog.columns and catalog.index.name != "feature_id":
            catalog = catalog.set_index("feature_id")

        return FeatureFrame(X=X, catalog=catalog, meta=meta)

    def put_frame(self, realization_id: str, frame: FeatureFrame) -> None:
        d = self._frame_dir(realization_id)
        _ensure_dir(d)

        x_path = d / "X.parquet"
        catalog_path = d / "catalog.parquet"
        meta_path = d / "meta.json"

        # Write payloads
        frame.X.to_parquet(x_path)
        frame.catalog.reset_index().to_parquet(catalog_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(frame.meta, f, ensure_ascii=False, indent=2, default=str)

        # Upsert index row
        con = self.conn()
        con.execute(
            """
            INSERT INTO frames(realization_id, x_path, catalog_path, meta_path, created_utc, n_rows, n_cols)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(realization_id) DO UPDATE SET
                x_path=excluded.x_path,
                catalog_path=excluded.catalog_path,
                meta_path=excluded.meta_path,
                created_utc=excluded.created_utc,
                n_rows=excluded.n_rows,
                n_cols=excluded.n_cols
        """,
            [
                realization_id,
                str(x_path),
                str(catalog_path),
                str(meta_path),
                _utc_now_iso(),
                int(frame.X.shape[0]),
                int(frame.X.shape[1]),
            ],
        )

    # ---------------------------
    # Fit states / artifacts
    # ---------------------------
    def _state_dir(self, state_id: str) -> Path:
        return self.root_path / "states" / state_id

    def put_state(self, state: FitState, payload: bytes) -> Artifact:
        d = self._state_dir(state.state_id)
        _ensure_dir(d)

        payload_path = d / "payload.bin"
        meta_path = d / "meta.json"

        payload_path.write_bytes(payload)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(state.meta, f, ensure_ascii=False, indent=2, default=str)

        con = self.conn()
        con.execute(
            """
            INSERT INTO states(state_id, payload_path, meta_path, created_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                payload_path=excluded.payload_path,
                meta_path=excluded.meta_path,
                created_utc=excluded.created_utc
        """,
            [state.state_id, str(payload_path), str(meta_path), _utc_now_iso()],
        )

        return Artifact(
            artifact_id=state.state_id,
            kind="fit_state",
            uri=str(payload_path),
            meta=state.meta,
        )

    def get_state(self, state_id: str) -> bytes:
        con = self.conn()
        row = con.execute(
            "SELECT payload_path FROM states WHERE state_id = ?",
            [state_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"FitState not found: {state_id}")
        return Path(row[0]).read_bytes()
