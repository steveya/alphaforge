import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITValidationError, PITValidationWarning
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def test_upsert_bool_aliases_for_strict_modes(tmp_path):
    pit = _make_accessor(tmp_path)
    bad = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "asof_utc": [pd.Timestamp("2025-01-10")],
            "value": [1.0],
        }
    )

    with pytest.raises(PITValidationError, match="naive_asof_rows"):
        pit.upsert_pit_observations(bad, strict=True)

    with pytest.warns(PITValidationWarning):
        pit.upsert_pit_observations(bad, strict=False)

    rows = pit.conn.execute("SELECT COUNT(*) FROM pit_observations WHERE series_key='GDP'").fetchone()[0]
    assert int(rows) == 1


def test_upsert_warn_mode_emits_typed_warning(tmp_path):
    pit = _make_accessor(tmp_path)
    bad = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2025-02-01")],
            "asof_utc": [pd.Timestamp("2025-01-10", tz="UTC")],
            "value": [1.0],
        }
    )

    with pytest.warns(PITValidationWarning):
        pit.upsert_pit_observations(bad, strict="warn")

    rows = pit.conn.execute("SELECT COUNT(*) FROM pit_observations WHERE series_key='GDP'").fetchone()[0]
    assert int(rows) == 1


def test_upsert_coerce_mode_repairs_rows_deterministically(tmp_path):
    pit = _make_accessor(tmp_path)
    incoming = pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-02-01"),
                "not-a-date",
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-01-10", tz="UTC"),
            ],
            "value": [1.0, 1.1, 2.0, 3.0],
        }
    )

    with pytest.warns(PITValidationWarning):
        pit.upsert_pit_observations(incoming, strict="coerce")

    got = pit.conn.execute(
        """
        SELECT obs_date, asof_utc, value
        FROM pit_observations
        WHERE series_key='GDP'
        ORDER BY obs_date, asof_utc
        """
    ).fetchdf()

    assert got.shape[0] == 1
    assert float(got.iloc[0]["value"]) == pytest.approx(1.1)
