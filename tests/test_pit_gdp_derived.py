import pandas as pd

from alphaforge.pit import (
    GDPC1_QOQ_SAAR_RF_REL_SERIES_KEY,
    GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY,
    PITAccessor,
    apply_gdpc1_qoq_saar_rf_rel,
    apply_gdpc1_qoq_saar_rf_rt,
)
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _gdp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDPC1"] * 7,
            "obs_date": [
                pd.Timestamp("2024-10-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-04-01"),
                pd.Timestamp("2025-04-01"),
                pd.Timestamp("2025-01-01"),
            ],
            "asof_utc": [
                pd.Timestamp("2024-11-01", tz="UTC"),
                pd.Timestamp("2025-02-01", tz="UTC"),
                pd.Timestamp("2025-03-01", tz="UTC"),
                pd.Timestamp("2025-04-01", tz="UTC"),
                pd.Timestamp("2025-05-01", tz="UTC"),
                pd.Timestamp("2025-06-01", tz="UTC"),
                pd.Timestamp("2025-05-15", tz="UTC"),
            ],
            "value": [100.0, 105.0, 106.0, 107.0, 110.0, 111.0, 120.0],
        }
    )


def test_get_snapshot_multi_matches_single_snapshot_union(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDP", "GDP", "CPI", "CPI"],
                "obs_date": [
                    pd.Timestamp("2024-12-31"),
                    pd.Timestamp("2025-03-31"),
                    pd.Timestamp("2024-12-31"),
                    pd.Timestamp("2025-03-31"),
                ],
                "asof_utc": [
                    pd.Timestamp("2025-01-10", tz="UTC"),
                    pd.Timestamp("2025-04-10", tz="UTC"),
                    pd.Timestamp("2025-01-12", tz="UTC"),
                    pd.Timestamp("2025-04-12", tz="UTC"),
                ],
                "value": [1.1, 2.2, 3.3, 4.4],
            }
        )
    )

    multi = pit.get_snapshot_multi(["GDP", "CPI"], pd.Timestamp("2025-05-01", tz="UTC"))
    expected = pd.concat(
        [
            pit.get_snapshot("GDP", pd.Timestamp("2025-05-01", tz="UTC")).rename("value").reset_index().assign(series_key="GDP"),
            pit.get_snapshot("CPI", pd.Timestamp("2025-05-01", tz="UTC")).rename("value").reset_index().assign(series_key="CPI"),
        ],
        ignore_index=True,
    )[["series_key", "obs_date", "value"]].sort_values(["series_key", "obs_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        multi[["series_key", "obs_date", "value"]],
        expected,
        check_dtype=False,
    )
    assert multi["source_asof_utc"].notna().all()


def test_get_revision_path_multi_matches_single_requests(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_gdp_df())

    requests = pd.DataFrame(
        {
            "request_id": ["a", "b"],
            "series_key": ["GDPC1", "GDPC1"],
            "obs_date": [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")],
        }
    )
    multi = pit.get_revision_path_multi(requests)
    single_a = pit.get_revision_path("GDPC1", pd.Timestamp("2025-01-01", tz="UTC")).assign(request_id="a")
    single_b = pit.get_revision_path("GDPC1", pd.Timestamp("2025-04-01", tz="UTC")).assign(request_id="b")
    expected = pd.concat([single_a, single_b], ignore_index=True)[
        ["request_id", "series_key", "obs_date", "asof_utc", "value", "revision_id"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(multi, expected, check_dtype=False)


def test_gdpc1_qoq_saar_rf_rt_uses_same_asof_pair(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_gdp_df())

    result = apply_gdpc1_qoq_saar_rf_rt(pit, overwrite=True)
    assert result.status == "success"
    got = pit.get_revision_path(
        GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY,
        pd.Timestamp("2025-04-01", tz="UTC"),
    )
    assert not got.empty
    first = got.loc[got["asof_utc"] == pd.Timestamp("2025-05-01", tz="UTC"), "value"].iloc[0]
    expected_first = 100.0 * ((110.0 / 107.0) ** 4.0 - 1.0)
    assert first == expected_first
    revised = got.loc[got["asof_utc"] == pd.Timestamp("2025-05-15", tz="UTC"), "value"].iloc[0]
    expected_revised = 100.0 * ((110.0 / 120.0) ** 4.0 - 1.0)
    assert revised == expected_revised


def test_gdpc1_qoq_saar_rf_rel_preserves_q_release_rank(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_gdp_df())

    result = apply_gdpc1_qoq_saar_rf_rel(pit, overwrite=True)
    assert result.status == "success"
    got = pit.get_revision_path(
        GDPC1_QOQ_SAAR_RF_REL_SERIES_KEY,
        pd.Timestamp("2025-04-01", tz="UTC"),
    )
    assert list(got["asof_utc"]) == [
        pd.Timestamp("2025-05-01", tz="UTC"),
        pd.Timestamp("2025-06-01", tz="UTC"),
    ]
    releases = pit.get_revision_path(
        "GDPC1",
        pd.Timestamp("2025-04-01", tz="UTC"),
    )
    assert list(got["asof_utc"]) == list(releases["asof_utc"])


def test_gdp_expression_graph_incremental_rerun_is_deterministic(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_gdp_df())

    first = apply_gdpc1_qoq_saar_rf_rt(pit, overwrite=True)
    assert first.status == "success"

    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDPC1"],
                "obs_date": [pd.Timestamp("2025-04-01")],
                "asof_utc": [pd.Timestamp("2025-07-01", tz="UTC")],
                "value": [112.0],
            }
        )
    )
    second = apply_gdpc1_qoq_saar_rf_rt(pit, incremental=True)
    assert second.incremental is True
    full = pit.get_revision_path(
        GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY,
        pd.Timestamp("2025-04-01", tz="UTC"),
    )
    assert pd.Timestamp("2025-07-01", tz="UTC") in set(full["asof_utc"])
