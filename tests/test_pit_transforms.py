import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.guards import ReleaseLagPolicy
from alphaforge.pit.transforms import PITTransformSpec, apply_obs_path_transform
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2024-02-10", tz="UTC"),
                pd.Timestamp("2024-03-10", tz="UTC"),
                pd.Timestamp("2024-03-15", tz="UTC"),
                pd.Timestamp("2024-04-15", tz="UTC"),
                pd.Timestamp("2024-04-20", tz="UTC"),
            ],
            "value": [1.0, 1.1, 2.0, 2.2, 3.0],
            "source": ["test"] * 5,
        }
    )


def test_apply_transform_resample_and_lineage(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_Q",
        op="resample",
        params={"rule": "Q", "agg": "last"},
        engine="auto",
    )
    result = pit.apply_transform(spec)

    assert result.rows_written > 0
    assert result.engine_used == "duckdb"
    assert result.transform_id.startswith("GDP_Q:")

    listed = pit.list_transforms("GDP_Q")
    assert not listed.empty
    assert listed.iloc[0]["op"] == "resample"

    out = pit.conn.execute(
        """
        SELECT obs_date, asof_utc, value
        FROM pit_observations
        WHERE series_key = ?
        ORDER BY asof_utc, obs_date
        """,
        ["GDP_Q"],
    ).fetchdf()
    assert not out.empty

    for asof in pd.to_datetime(out["asof_utc"], utc=True).unique():
        snap = pit.get_snapshot("GDP", asof)
        expected = apply_obs_path_transform(snap, spec).dropna()
        got = out[pd.to_datetime(out["asof_utc"], utc=True) == asof]
        got_series = pd.Series(
            got["value"].to_numpy(),
            index=pd.to_datetime(got["obs_date"], utc=True),
        ).sort_index()
        pd.testing.assert_series_equal(
            got_series,
            expected.reindex(got_series.index),
            check_names=False,
        )


def test_apply_transform_path_apply_requires_udf_name(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_custom",
        op="path_apply",
        params={"func": lambda s: s * 2.0},
    )

    try:
        pit.apply_transform(spec)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "udf_name" in str(exc)


def test_apply_transform_with_lag_policy(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff",
        op="diff",
        params={"periods": 1},
    )
    no_lag = pit.apply_transform(spec, overwrite=True)

    policy = ReleaseLagPolicy(default_lag=pd.Timedelta(days=40))
    lagged = pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP",
            output_series_key="GDP_diff_lagged",
            op="diff",
            params={"periods": 1},
        ),
        lag_policy=policy,
        overwrite=True,
    )

    assert lagged.rows_written <= no_lag.rows_written


def test_revision_path_diff_transform(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_rev_diff",
        axis="revision_path",
        op="diff",
        params={"periods": 1},
    )
    result = pit.apply_transform(spec, overwrite=True)
    assert result.rows_written > 0

    out = pit.conn.execute(
        """
        SELECT obs_date, asof_utc, value
        FROM pit_observations
        WHERE series_key = ?
        ORDER BY obs_date, asof_utc
        """,
        ["GDP_rev_diff"],
    ).fetchdf()
    assert not out.empty

    # For 2024-01-31, value changed 1.0 -> 1.1, so diff at second asof is 0.1
    jan = out[pd.to_datetime(out["obs_date"], utc=True) == pd.Timestamp("2024-01-31", tz="UTC")]
    assert jan.shape[0] == 1
    assert jan.iloc[0]["value"] == pytest.approx(0.1)


def test_invalid_axis_op_combination_raises_deterministic_error(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_bad",
        axis="revision_path",
        op="resample",
        params={"rule": "Q", "agg": "last"},
    )

    with pytest.raises(ValueError, match="Supported combinations"):
        pit.apply_transform(spec)


def test_obs_path_causality_violation_rejected(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff_future",
        op="diff",
        params={"periods": 1},
    )
    # Negative lag means effective source_asof > output asof and must be rejected.
    policy = ReleaseLagPolicy(default_lag=-pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="Causality violation"):
        pit.apply_transform(spec, lag_policy=policy)
