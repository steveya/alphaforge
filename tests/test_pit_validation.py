import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError, PITValidationError
from alphaforge.pit.validation import validate_pit_observations
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def test_validate_pit_observations_report_counts():
    df = pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-03-01", tz="UTC"),
            ],
            "value": [1.0, 1.0, 2.0],
            "release_time_utc": [
                pd.Timestamp("2025-02-10"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-03-01", tz="UTC"),
            ],
        }
    )

    report = validate_pit_observations(df)
    assert report.rows == 3
    assert report.duplicate_key_rows == 2
    assert report.future_rows == 1
    assert report.naive_release_rows == 1
    assert report.has_errors


def test_upsert_strict_rejects_timezone_issues(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "asof_utc": [pd.Timestamp("2025-01-10")],
            "value": [1.0],
        }
    )

    with pytest.raises(PITValidationError, match="naive_asof_rows"):
        pit.upsert_pit_observations(df, strict=True)


def test_upsert_strict_rejects_future_rows(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2025-02-01")],
            "asof_utc": [pd.Timestamp("2025-01-10", tz="UTC")],
            "value": [1.0],
        }
    )

    with pytest.raises(PITValidationError, match="future_rows"):
        pit.upsert_pit_observations(df, strict=True)


def test_upsert_strict_rejects_duplicate_keys(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": ["GDP", "GDP"],
            "obs_date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-01-10", tz="UTC"),
            ],
            "value": [1.0, 1.1],
        }
    )

    with pytest.raises(PITValidationError, match="duplicate_key_rows"):
        pit.upsert_pit_observations(df, strict=True)


def test_upsert_strict_rejects_null_required_rows(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": [None],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "asof_utc": [pd.Timestamp("2025-01-10", tz="UTC")],
            "value": [1.0],
        }
    )

    with pytest.raises(PITValidationError, match="null_required_rows"):
        pit.upsert_pit_observations(df, strict=True)


def test_upsert_missing_columns_is_contract_error(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "value": [1.0],
        }
    )

    with pytest.raises(PITContractError, match="Missing required columns"):
        pit.upsert_pit_observations(df, strict=True)


def test_upsert_non_strict_allows_timezone_issue(tmp_path):
    pit = _make_accessor(tmp_path)
    df = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "asof_utc": [pd.Timestamp("2025-01-10")],
            "value": [1.0],
        }
    )

    pit.upsert_pit_observations(df, strict=False)
    out = pit.conn.execute(
        "SELECT COUNT(*) FROM pit_observations WHERE series_key = ?",
        ["GDP"],
    ).fetchone()[0]
    assert out == 1

