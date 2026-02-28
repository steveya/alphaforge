import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import (
    PITCausalityError,
    PITEngineError,
    PITExperimentalFeatureError,
    PITUnsupportedOperationError,
    PITValidationError,
)
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


def _sorted_obs_asof_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["obs_date", "asof_utc", "value"]].copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"], utc=True)
    out["asof_utc"] = pd.to_datetime(out["asof_utc"], utc=True)
    return out.sort_values(["obs_date", "asof_utc"]).reset_index(drop=True)


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
    assert result.engine_requested == "auto"
    assert result.fallback_reason is None
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


def test_apply_transform_accepts_mapping_spec(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    result = pit.apply_transform(
        {
            "input_series_key": "GDP",
            "output_series_key": "GDP_lag1",
            "op": "lag",
            "params": {"periods": 1},
        },
        overwrite=True,
    )

    assert result.rows_written > 0


def test_transform_hash_stable_for_semantically_equivalent_params():
    a = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_Q",
        op="resample",
        params={"rule": "Q", "agg": "LAST"},
    )
    b = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_Q",
        op="resample",
        params={"rule": "QE", "agg": "last"},
    )
    assert a.spec_hash() == b.spec_hash()


def test_apply_transform_unknown_param_rejected(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_bad",
        op="lag",
        params={"periods": 1, "foo": "bar"},
    )

    with pytest.raises(PITValidationError, match="Unknown params"):
        pit.apply_transform(spec)


def test_apply_transform_path_apply_requires_udf_name(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_custom",
        op="path_apply",
        params={"func": lambda s: s * 2.0},
    )

    with pytest.raises(PITValidationError, match="udf_name"):
        pit.apply_transform(spec)


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


def test_revision_path_requires_explicit_opt_in(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_rev_diff",
        axis="revision_path",
        op="diff",
        params={"periods": 1},
    )

    with pytest.raises(PITExperimentalFeatureError, match="allow_experimental"):
        pit.apply_transform(spec)


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
    result = pit.apply_transform(spec, overwrite=True, allow_experimental=True)
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

    with pytest.raises(PITUnsupportedOperationError, match="Supported combinations"):
        pit.apply_transform(spec, allow_experimental=True)


def test_obs_path_causality_violation_rejected(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff_future",
        op="diff",
        params={"periods": 1},
    )
    policy = ReleaseLagPolicy(default_lag=-pd.Timedelta(days=1))
    with pytest.raises(PITCausalityError, match="Causality violation"):
        pit.apply_transform(spec, lag_policy=policy)


def test_engine_contract_duckdb_error_and_fallback(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff_engine",
        op="path_apply",
        params={"udf_name": "identity", "func": lambda s: s},
        engine="duckdb",
    )

    with pytest.raises(PITEngineError, match="not supported"):
        pit.apply_transform(spec)

    result = pit.apply_transform(
        spec,
        on_engine_mismatch="fallback",
        overwrite=True,
    )
    assert result.engine_requested == "duckdb"
    assert result.engine_used == "python"
    assert result.fallback_reason == "duckdb_unsupported_for_spec"


def test_engine_auto_path_apply_uses_python(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    result = pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP",
            output_series_key="GDP_identity_auto",
            op="path_apply",
            params={"udf_name": "identity", "func": lambda s: s},
            engine="auto",
        ),
        persist=False,
    )

    assert result.engine_requested == "auto"
    assert result.engine_used == "python"
    assert result.fallback_reason is None


def test_engine_auto_revision_path_uses_duckdb_for_supported_op(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    result = pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP",
            output_series_key="GDP_revision_auto",
            axis="revision_path",
            op="diff",
            params={"periods": 1},
            engine="auto",
        ),
        allow_experimental=True,
        persist=False,
    )

    assert result.engine_requested == "auto"
    assert result.engine_used == "duckdb"
    assert result.fallback_reason is None


@pytest.mark.parametrize(
    ("op", "params"),
    [
        ("lag", {"periods": 1}),
        ("diff", {"periods": 1}),
        ("rolling", {"window": 2, "min_periods": 1, "agg": "mean"}),
        ("expanding", {"min_periods": 1, "agg": "mean"}),
    ],
)
def test_python_duckdb_parity_for_supported_path_ops(tmp_path, op, params):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    python_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key=f"GDP_{op}_py",
        op=op,
        params=params,
        engine="python",
    )
    duckdb_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key=f"GDP_{op}_ddb",
        op=op,
        params=params,
        engine="duckdb",
    )

    py_preview = pit.preview_transform(python_spec)
    ddb_preview = pit.preview_transform(duckdb_spec)

    py_cmp = _sorted_obs_asof_values(py_preview)
    ddb_cmp = _sorted_obs_asof_values(ddb_preview)

    pd.testing.assert_frame_equal(py_cmp, ddb_cmp, check_dtype=False, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("op", "params"),
    [
        ("lag", {"periods": 1}),
        ("diff", {"periods": 1}),
        ("rolling", {"window": 2, "min_periods": 1, "agg": "mean"}),
        ("expanding", {"min_periods": 1, "agg": "mean"}),
    ],
)
def test_python_duckdb_parity_for_revision_path_ops(tmp_path, op, params):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    python_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key=f"GDP_revision_{op}_py",
        axis="revision_path",
        op=op,
        params=params,
        engine="python",
    )
    duckdb_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key=f"GDP_revision_{op}_ddb",
        axis="revision_path",
        op=op,
        params=params,
        engine="duckdb",
    )

    py_preview = pit.preview_transform(python_spec, allow_experimental=True)
    ddb_preview = pit.preview_transform(duckdb_spec, allow_experimental=True)

    py_cmp = _sorted_obs_asof_values(py_preview)
    ddb_cmp = _sorted_obs_asof_values(ddb_preview)
    pd.testing.assert_frame_equal(py_cmp, ddb_cmp, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_preview_transform_matches_apply_rows(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_preview",
        op="diff",
        params={"periods": 1},
    )

    preview = pit.preview_transform(spec)
    result = pit.apply_transform(spec, overwrite=True)
    assert result.rows_written == len(preview)

    got = pit.conn.execute(
        """
        SELECT series_key, obs_date, asof_utc, value
        FROM pit_observations
        WHERE series_key = ?
        ORDER BY obs_date, asof_utc
        """,
        ["GDP_preview"],
    ).fetchdf()

    preview_sorted = preview[["series_key", "obs_date", "asof_utc", "value"]].copy()
    preview_sorted["obs_date"] = pd.to_datetime(preview_sorted["obs_date"], utc=True)
    preview_sorted["asof_utc"] = pd.to_datetime(preview_sorted["asof_utc"], utc=True)
    preview_sorted = preview_sorted.sort_values(["obs_date", "asof_utc"]).reset_index(drop=True)

    got["obs_date"] = pd.to_datetime(got["obs_date"], utc=True)
    got["asof_utc"] = pd.to_datetime(got["asof_utc"], utc=True)
    got = got.sort_values(["obs_date", "asof_utc"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(preview_sorted, got, check_dtype=False)


def test_python_duckdb_parity_for_resample(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    py_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_resample_py",
        op="resample",
        params={"rule": "Q", "agg": "last"},
        engine="python",
    )
    ddb_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_resample_ddb",
        op="resample",
        params={"rule": "Q", "agg": "last"},
        engine="duckdb",
    )

    py_preview = pit.preview_transform(py_spec)
    ddb_preview = pit.preview_transform(ddb_spec)

    py_cmp = _sorted_obs_asof_values(py_preview)
    ddb_cmp = _sorted_obs_asof_values(ddb_preview)

    pd.testing.assert_frame_equal(py_cmp, ddb_cmp, check_dtype=False, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        {"agg": "last"},
        {"rule": "Q", "agg": "sum"},
    ],
)
def test_python_duckdb_parity_for_aggregate(tmp_path, params):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    py_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_aggregate_py",
        op="aggregate",
        params=params,
        engine="python",
    )
    ddb_spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_aggregate_ddb",
        op="aggregate",
        params=params,
        engine="duckdb",
    )

    py_preview = pit.preview_transform(py_spec)
    ddb_preview = pit.preview_transform(ddb_spec)
    py_cmp = _sorted_obs_asof_values(py_preview)
    ddb_cmp = _sorted_obs_asof_values(ddb_preview)

    pd.testing.assert_frame_equal(py_cmp, ddb_cmp, check_dtype=False, atol=1e-12, rtol=1e-12)
