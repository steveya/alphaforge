from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge import Query
from alphaforge.futures import (
    FirstRateFuturesConfig,
    FirstRateFuturesLoader,
    build_first_rate_futures_context,
)


def _write_contract(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_roll_source_dir(base: Path) -> tuple[Path, Path]:
    source_dir = base / "source"
    artifact_root = base / "artifacts"
    source_dir.mkdir()

    rows_m = [
        "2024-06-16 18:00:00,100,101,99,100,10",
        "2024-06-16 18:05:00,100,101,100,101,10",
        "2024-06-17 16:55:00,101,102,100,102,10",
        "2024-06-17 18:00:00,102,103,101,103,50",
        "2024-06-17 18:05:00,103,104,102,104,50",
        "2024-06-18 16:55:00,104,105,103,105,50",
        "2024-06-18 18:00:00,105,106,104,106,20",
        "2024-06-18 18:05:00,106,107,105,107,20",
        "2024-06-19 16:55:00,107,108,106,108,20",
    ]
    rows_u = [
        "2024-06-17 18:00:00,200,201,199,200,60",
        "2024-06-17 18:05:00,200,202,200,201,60",
        "2024-06-18 16:55:00,201,203,201,202,60",
        "2024-06-18 18:00:00,202,204,202,203,70",
        "2024-06-18 18:05:00,203,205,203,204,70",
        "2024-06-19 16:55:00,204,206,204,205,70",
        "2024-06-19 18:00:00,205,207,205,206,80",
        "2024-06-19 18:05:00,206,208,206,207,80",
        "2024-06-20 16:55:00,207,209,207,208,80",
    ]

    _write_contract(source_dir / "ES_M24_5min.txt", rows_m)
    _write_contract(source_dir / "ES_U24_5min.txt", rows_u)
    return source_dir, artifact_root


def test_loader_rejects_empty_directory(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=tmp_path / "artifacts")
    )

    with pytest.raises(ValueError, match="No contract files found"):
        loader.ingest()


def test_loader_rejects_nested_directories(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=tmp_path / "artifacts")
    )

    with pytest.raises(ValueError, match="Expected a flat source_dir"):
        loader.ingest()


def test_loader_rejects_invalid_filename(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_contract(source_dir / "bad_name.txt", ["2024-06-16 18:00:00,1,1,1,1,1"])
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=tmp_path / "artifacts")
    )

    with pytest.raises(ValueError, match="Invalid First Rate contract filenames"):
        loader.ingest()


def test_loader_rejects_unsupported_root(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_contract(
        source_dir / "TOOLONG_H24_5min.txt",
        ["2024-06-16 18:00:00,1,1,1,1,1"],
    )
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=tmp_path / "artifacts")
    )

    with pytest.raises(ValueError, match="Unsupported futures roots"):
        loader.ingest()


def test_loader_converts_eastern_to_utc_and_assigns_session_date(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_contract(
        source_dir / "ES_H24_5min.txt",
        ["2024-03-10 18:00:00,1,2,1,2,3"],
    )
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=tmp_path / "artifacts")
    )

    loader.ingest()
    sparse = pd.read_parquet(tmp_path / "artifacts" / "contract_5m_sparse" / "data.parquet")

    assert sparse.loc[0, "bar_start_utc"] == pd.Timestamp("2024-03-10 22:00:00+00:00")
    assert sparse.loc[0, "available_at_utc"] == pd.Timestamp("2024-03-10 22:05:00+00:00")
    assert sparse.loc[0, "session_date"] == pd.Timestamp("2024-03-11")


def test_loader_builds_dense_eod_and_continuous_outputs(tmp_path) -> None:
    source_dir, artifact_root = _make_roll_source_dir(tmp_path)
    loader = FirstRateFuturesLoader(
        FirstRateFuturesConfig(source_dir=source_dir, artifact_root=artifact_root)
    )

    loader.ingest()

    dense = pd.read_parquet(artifact_root / "contract_5m_dense" / "data.parquet")
    gap_row = dense[
        (dense["contract_id"] == "ES_M24")
        & (dense["bar_start_utc"] == pd.Timestamp("2024-06-16 22:10:00+00:00"))
    ].iloc[0]
    assert bool(gap_row["is_synthetic_bar"]) is True
    assert float(gap_row["volume"]) == 0.0
    assert float(gap_row["open"]) == 101.0
    assert float(gap_row["close"]) == 101.0

    contract_eod = pd.read_parquet(artifact_root / "contract_eod" / "data.parquet")
    first_session = contract_eod[
        (contract_eod["contract_id"] == "ES_M24")
        & (contract_eod["session_date"] == pd.Timestamp("2024-06-17"))
    ].iloc[0]
    assert float(first_session["open"]) == 100.0
    assert float(first_session["close"]) == 102.0
    assert int(first_session["bar_count"]) == 276
    assert int(first_session["observed_bar_count"]) == 3
    assert bool(first_session["is_partial_session"]) is False

    schedule = pd.read_parquet(artifact_root / "roll_schedules" / "data.parquet")
    assert schedule["active_contract_id"].tolist() == ["ES_M24", "ES_U24"]
    assert schedule["start_session_date"].tolist() == [
        pd.Timestamp("2024-06-17"),
        pd.Timestamp("2024-06-20"),
    ]

    continuous_exec = pd.read_parquet(
        artifact_root / "continuous_5m_execution" / "data.parquet"
    )
    first_u24_row = continuous_exec[
        continuous_exec["active_contract_id"] == "ES_U24"
    ].sort_values("available_at_utc").iloc[0]
    assert bool(first_u24_row["roll_flag"]) is True
    assert first_u24_row["session_date"] == pd.Timestamp("2024-06-20")

    continuous_eod = pd.read_parquet(
        artifact_root / "continuous_eod_research" / "data.parquet"
    )
    assert continuous_eod["active_contract_id"].tolist() == [
        "ES_M24",
        "ES_M24",
        "ES_M24",
        "ES_U24",
    ]
    assert float(continuous_eod.iloc[-1]["adjustment_factor"]) > 1.0
    assert float(continuous_eod.iloc[0]["cumulative_adjustment_factor"]) > 1.0


def test_context_adapter_reads_manifest_and_fetches_data(tmp_path) -> None:
    source_dir, artifact_root = _make_roll_source_dir(tmp_path)
    cfg = FirstRateFuturesConfig(source_dir=source_dir, artifact_root=artifact_root)
    FirstRateFuturesLoader(cfg).ingest()

    ctx = build_first_rate_futures_context(cfg)
    adapter = ctx.adapters["first_rate_futures"]

    assert adapter.list_entities("futures.continuous_eod_research") == ["ES"]

    result = ctx.fetch(
        Query(
            table="futures.continuous_eod_research",
            columns=["close", "active_contract_id"],
            entities=["ES"],
            start="2024-06-17T00:00:00Z",
            end="2024-06-21T00:00:00Z",
        )
    )

    assert result.dataset == "futures.continuous_eod_research"
    assert result.source == "first_rate_futures"
    assert result.is_pit is False
    assert result.data["series_key"].tolist() == ["ES", "ES", "ES", "ES"]
    assert result.data["active_contract_id"].tolist()[-1] == "ES_U24"
