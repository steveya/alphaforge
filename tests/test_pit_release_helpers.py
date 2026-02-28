import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.ref_period import RefFreq


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_release_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-03-10", tz="UTC"),
            ],
            "value": [1.0, 1.1, 1.2],
            "revision_id": ["v1", "v2", "v3"],
        }
    )


def test_list_release_stream_has_rank_flags_and_asof_cutoff(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_release_df())

    stream = pit.list_release_stream("GDP", "2024Q4")
    assert stream["release_rank"].tolist() == [1, 2, 3]
    assert bool(stream.iloc[0]["is_first"]) is True
    assert bool(stream.iloc[-1]["is_latest"]) is True

    cutoff = pit.list_release_stream(
        "GDP",
        "2024Q4",
        asof=pd.Timestamp("2025-02-15", tz="UTC"),
    )
    assert cutoff["release_rank"].tolist() == [1, 2]


def test_resolve_release_policies(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_release_df())

    first = pit.resolve_release("GDP", "2024Q4", policy="first")
    latest = pit.resolve_release("GDP", "2024Q4", policy="latest")
    ranked = pit.resolve_release("GDP", "2024Q4", policy={"mode": "rank", "rank": 2})
    horizon = pit.resolve_release(
        "GDP",
        "2024Q4",
        policy={"mode": "horizon", "horizon": pd.Timedelta(days=40)},
    )

    assert first is not None and first.release_rank == 1
    assert latest is not None and latest.release_rank == 3
    assert ranked is not None and ranked.release_rank == 2
    assert horizon is not None and horizon.release_rank == 1


def test_release_helpers_validate_ref_frequency(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_release_df())

    with pytest.raises(PITContractError, match="frequency"):
        pit.list_release_stream("GDP", "2024Q4", freq=RefFreq.M)
