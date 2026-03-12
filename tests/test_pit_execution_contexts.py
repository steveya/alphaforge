import json

import pandas as pd
import pytest

from alphaforge.pit import (
    PITAccessor,
    PITContractError,
    PITTapeSpec,
    PITTransformSpec,
    SnapshotSeriesSpec,
    build_snapshot_tape,
    iter_purged_kfold_folds,
    iter_walk_forward_folds,
)
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _ratio_splice_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["OLD"] * 4 + ["NEW"] * 4 + ["NEW"] * 4,
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
                pd.Timestamp("2024-05-31"),
                pd.Timestamp("2024-06-30"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
                pd.Timestamp("2024-05-31"),
                pd.Timestamp("2024-06-30"),
            ],
            "asof_utc": [pd.Timestamp("2024-07-01", tz="UTC")] * 4
            + [pd.Timestamp("2024-07-15", tz="UTC")] * 4
            + [pd.Timestamp("2024-08-15", tz="UTC")] * 4,
            "value": [
                10.0,
                11.0,
                12.0,
                13.0,
                24.0,
                26.0,
                28.0,
                30.0,
                24.0,
                28.0,
                30.0,
                32.0,
            ],
        }
    )


def _additive_splice_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["OLD"] * 4 + ["NEW"] * 4,
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
                pd.Timestamp("2024-05-31"),
                pd.Timestamp("2024-06-30"),
            ],
            "asof_utc": [pd.Timestamp("2024-07-01", tz="UTC")] * 4
            + [pd.Timestamp("2024-07-15", tz="UTC")] * 4,
            "value": [100.0, 110.0, 120.0, 130.0, 80.0, 95.0, 105.0, 115.0],
        }
    )


def _no_overlap_splice_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["OLD"] * 2 + ["NEW"] * 2,
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-30"),
            ],
            "asof_utc": [pd.Timestamp("2024-05-15", tz="UTC")] * 4,
            "value": [10.0, 11.0, 50.0, 55.0],
        }
    )


def _tape_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-04-10", tz="UTC"),
            ],
            "value": [1.0, 1.2, 2.0],
        }
    )


def test_splice_ratio_transform_recalibrates_with_revised_overlap(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_ratio_splice_df())

    spec = PITTransformSpec(
        input_series_key="OLD",
        output_series_key="OLD_NEW_ratio_splice",
        op="splice",
        params={
            "right_series_key": "NEW",
            "adjustment": "ratio",
            "join": "outer",
        },
    )

    result = pit.apply_transform(spec, overwrite=True)
    assert result.engine_used == "python"

    out = pit.conn.execute(
        """
        SELECT obs_date, asof_utc, value, meta_json
        FROM pit_observations
        WHERE series_key = ?
        ORDER BY asof_utc, obs_date
        """,
        ["OLD_NEW_ratio_splice"],
    ).fetchdf()
    assert not out.empty

    asof_0715 = out[pd.to_datetime(out["asof_utc"], utc=True) == pd.Timestamp("2024-07-15", tz="UTC")]
    asof_0815 = out[pd.to_datetime(out["asof_utc"], utc=True) == pd.Timestamp("2024-08-15", tz="UTC")]

    got_0715 = pd.Series(
        asof_0715["value"].to_numpy(),
        index=pd.to_datetime(asof_0715["obs_date"], utc=True),
    ).sort_index()
    got_0815 = pd.Series(
        asof_0815["value"].to_numpy(),
        index=pd.to_datetime(asof_0815["obs_date"], utc=True),
    ).sort_index()

    expected_0515 = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-29", tz="UTC"),
                pd.Timestamp("2024-03-31", tz="UTC"),
                pd.Timestamp("2024-04-30", tz="UTC"),
                pd.Timestamp("2024-05-31", tz="UTC"),
                pd.Timestamp("2024-06-30", tz="UTC"),
            ]
        ),
    )
    expected_0615 = pd.Series(
        [10.0, 11.0, 11.142857142857142, 13.0, 13.928571428571429, 14.857142857142858],
        index=expected_0515.index,
    )

    pd.testing.assert_series_equal(
        got_0715,
        expected_0515,
        check_names=False,
        check_index_type=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_series_equal(
        got_0815,
        expected_0615,
        check_names=False,
        check_index_type=False,
        atol=1e-12,
        rtol=1e-12,
    )

    meta = {
        pd.Timestamp(row.obs_date, tz="UTC"): json.loads(row.meta_json)
        for row in asof_0715.itertuples(index=False)
    }
    future_row = meta[pd.Timestamp("2024-05-31", tz="UTC")]
    assert future_row["selected_input_series_key"] == "NEW"
    assert future_row["splice_anchor_obs_date_utc"] == "2024-04-30T00:00:00+00:00"
    assert future_row["splice_scale"] == pytest.approx(0.5)
    assert future_row["splice_left_weight"] == pytest.approx(0.0)
    assert future_row["splice_right_weight"] == pytest.approx(1.0)
    assert future_row["splice_right_input_asof_utc"] == "2024-07-15T00:00:00+00:00"


def test_splice_additive_transition_blends_left_and_right(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_additive_splice_df())

    preview = pit.preview_transform(
        PITTransformSpec(
            input_series_key="OLD",
            output_series_key="OLD_NEW_add_splice",
            op="splice",
            params={
                "right_series_key": "NEW",
                "adjustment": "add",
                "transition_periods": 2,
                "join": "outer",
            },
        )
    )

    asof_0715 = preview[pd.to_datetime(preview["asof_utc"], utc=True) == pd.Timestamp("2024-07-15", tz="UTC")]
    got = pd.Series(
        asof_0715["value"].to_numpy(),
        index=pd.to_datetime(asof_0715["obs_date"], utc=True),
    ).sort_index()
    expected = pd.Series(
        [100.0, 110.0, 118.33333333333333, 130.0, 140.0, 150.0],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-29", tz="UTC"),
                pd.Timestamp("2024-03-31", tz="UTC"),
                pd.Timestamp("2024-04-30", tz="UTC"),
                pd.Timestamp("2024-05-31", tz="UTC"),
                pd.Timestamp("2024-06-30", tz="UTC"),
            ]
        ),
    )

    pd.testing.assert_series_equal(
        got,
        expected,
        check_names=False,
        check_index_type=False,
        atol=1e-12,
        rtol=1e-12,
    )

    transition_row = json.loads(asof_0715.iloc[2]["meta_json"])
    assert transition_row["splice_state"] == "transition"
    assert transition_row["splice_left_weight"] == pytest.approx(2.0 / 3.0)
    assert transition_row["splice_right_weight"] == pytest.approx(1.0 / 3.0)
    assert transition_row["splice_offset"] == pytest.approx(35.0)


def test_splice_without_overlap_omits_uncalibrated_handoff_rows(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_no_overlap_splice_df())

    preview = pit.preview_transform(
        PITTransformSpec(
            input_series_key="OLD",
            output_series_key="OLD_NEW_uncalibrated",
            op="splice",
            params={
                "right_series_key": "NEW",
                "adjustment": "ratio",
            },
        )
    )

    got = pd.Series(
        preview["value"].to_numpy(),
        index=pd.to_datetime(preview["obs_date"], utc=True),
    ).sort_index()
    expected = pd.Series(
        [10.0, 11.0],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-29", tz="UTC"),
            ]
        ),
    )
    pd.testing.assert_series_equal(got, expected, check_names=False, check_index_type=False)


def test_iter_walk_forward_folds_uses_sorted_asof_grid_and_purge():
    asof_grid = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-05", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-03", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
            pd.Timestamp("2025-01-04", tz="UTC"),
        ]
    )

    folds = list(
        iter_walk_forward_folds(
            asof_grid,
            train_size=2,
            validation_size=1,
            step=1,
            purge=1,
        )
    )

    assert len(folds) == 2
    assert folds[0].fold_mode == "walk_forward"
    assert folds[0].train_asofs == (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-02", tz="UTC"),
    )
    assert folds[0].validation_asofs == (pd.Timestamp("2025-01-04", tz="UTC"),)
    assert folds[1].train_asofs == (
        pd.Timestamp("2025-01-02", tz="UTC"),
        pd.Timestamp("2025-01-03", tz="UTC"),
    )
    assert folds[1].validation_asofs == (pd.Timestamp("2025-01-05", tz="UTC"),)


def test_iter_purged_kfold_folds_respects_purge_and_embargo():
    asof_grid = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    folds = list(iter_purged_kfold_folds(asof_grid, n_splits=3, purge=1, embargo=1))

    assert len(folds) == 3
    assert folds[0].validation_asofs == (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-02", tz="UTC"),
    )
    assert folds[0].train_asofs == (
        pd.Timestamp("2025-01-04", tz="UTC"),
        pd.Timestamp("2025-01-05", tz="UTC"),
        pd.Timestamp("2025-01-06", tz="UTC"),
    )
    assert folds[1].validation_asofs == (
        pd.Timestamp("2025-01-03", tz="UTC"),
        pd.Timestamp("2025-01-04", tz="UTC"),
    )
    assert folds[1].train_asofs == (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-06", tz="UTC"),
    )


def test_build_snapshot_tape_filtered_vs_smoothed_and_release_policy(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_tape_df())

    filtered = build_snapshot_tape(
        pit,
        PITTapeSpec(
            series_specs=(
                SnapshotSeriesSpec(series_key="GDP", alias="gdp_latest"),
                SnapshotSeriesSpec(series_key="GDP", alias="gdp_first", release_policy="first"),
            ),
            step_asofs=(
                pd.Timestamp("2025-01-15", tz="UTC"),
                pd.Timestamp("2025-03-15", tz="UTC"),
            ),
            mode="filtered",
        ),
    )
    assert set(filtered["sequence_mode"]) == {"filtered"}

    latest_step1 = filtered[
        (filtered["series_alias"] == "gdp_latest")
        & (filtered["step_asof_utc"] == pd.Timestamp("2025-01-15", tz="UTC"))
        & (filtered["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
    ].iloc[0]
    latest_step2 = filtered[
        (filtered["series_alias"] == "gdp_latest")
        & (filtered["step_asof_utc"] == pd.Timestamp("2025-03-15", tz="UTC"))
        & (filtered["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
    ].iloc[0]
    first_step2 = filtered[
        (filtered["series_alias"] == "gdp_first")
        & (filtered["step_asof_utc"] == pd.Timestamp("2025-03-15", tz="UTC"))
        & (filtered["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
    ].iloc[0]

    assert latest_step1["value"] == pytest.approx(1.0)
    assert latest_step1["source_asof_utc"] == pd.Timestamp("2025-01-10", tz="UTC")
    assert latest_step2["value"] == pytest.approx(1.2)
    assert latest_step2["source_asof_utc"] == pd.Timestamp("2025-02-10", tz="UTC")
    assert first_step2["value"] == pytest.approx(1.0)
    assert first_step2["source_asof_utc"] == pd.Timestamp("2025-01-10", tz="UTC")

    with pytest.raises(PITContractError, match="allow_research=True"):
        build_snapshot_tape(
            pit,
            {
                "series_specs": [{"series_key": "GDP", "alias": "gdp_latest"}],
                "step_asofs": [
                    pd.Timestamp("2025-01-15", tz="UTC"),
                    pd.Timestamp("2025-03-15", tz="UTC"),
                ],
                "mode": "smoothed_research",
            },
        )

    smoothed = build_snapshot_tape(
        pit,
        {
            "series_specs": [{"series_key": "GDP", "alias": "gdp_latest"}],
            "step_asofs": [
                pd.Timestamp("2025-01-15", tz="UTC"),
                pd.Timestamp("2025-03-15", tz="UTC"),
            ],
            "mode": "smoothed_research",
        },
        allow_research=True,
    )
    assert set(smoothed["sequence_mode"]) == {"smoothed_research"}
    step1_smoothed = smoothed[
        (smoothed["step_asof_utc"] == pd.Timestamp("2025-01-15", tz="UTC"))
        & (smoothed["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
    ].iloc[0]
    assert step1_smoothed["value"] == pytest.approx(1.2)
    assert step1_smoothed["source_asof_utc"] == pd.Timestamp("2025-02-10", tz="UTC")
    assert step1_smoothed["materialized_asof_utc"] == pd.Timestamp("2025-03-15", tz="UTC")
