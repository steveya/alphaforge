import pandas as pd
import pytest

from alphaforge.data.context import DataContext
from alphaforge.data.pit_source import PITDataSource
from alphaforge.data.query import Query
from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError
from alphaforge.pit.pipelines import PITPipelineSpec, PITPipelineStep
from alphaforge.pit.transforms import PITTransformSpec
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


def _sample_pipeline() -> PITPipelineSpec:
    return PITPipelineSpec(
        pipeline_id="macro/gdp_lag_diff",
        description="Lag + diff pipeline for GDP",
        steps=(
            PITPipelineStep(
                name="lag",
                spec=PITTransformSpec(
                    input_series_key="GDP",
                    output_series_key="GDP_lag1",
                    op="lag",
                    params={"periods": 1},
                ),
            ),
            PITPipelineStep(
                name="diff",
                spec=PITTransformSpec(
                    input_series_key="GDP_lag1",
                    output_series_key="GDP_lag1_diff",
                    op="diff",
                    params={"periods": 1},
                ),
                depends_on=("lag",),
            ),
        ),
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["series_key", "obs_date", "asof_utc", "value"]].copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"], utc=True)
    out["asof_utc"] = pd.to_datetime(out["asof_utc"], utc=True)
    return out.sort_values(["series_key", "obs_date", "asof_utc"]).reset_index(drop=True)


def test_preview_pipeline_apply_parity_final_step(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = _sample_pipeline()
    preview = pit.preview_pipeline(spec, overwrite=True)

    persisted = pit.conn.execute(
        "SELECT COUNT(*) FROM pit_observations WHERE series_key IN ('GDP_lag1', 'GDP_lag1_diff')"
    ).fetchone()[0]
    assert int(persisted) == 0

    result = pit.apply_pipeline(spec, overwrite=True)
    assert result.status == "success"
    assert result.pipeline_id == "macro/gdp_lag_diff"
    assert len(result.step_results) == 2
    assert result.rows_written == sum(step.rows_written for step in result.step_results)

    got = pit.conn.execute(
        """
        SELECT series_key, obs_date, asof_utc, value
        FROM pit_observations
        WHERE series_key = ?
        ORDER BY obs_date, asof_utc
        """,
        ["GDP_lag1_diff"],
    ).fetchdf()
    pd.testing.assert_frame_equal(_sorted(preview), _sorted(got), check_dtype=False)


def test_preview_pipeline_include_intermediate(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    preview = pit.preview_pipeline(_sample_pipeline(), include_intermediate=True, overwrite=True)
    assert not preview.empty
    assert set(preview["step_name"].dropna().tolist()) == {"lag", "diff"}


def test_pipeline_incremental_anchor_and_run_listing(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    spec = _sample_pipeline()
    first = pit.apply_pipeline(spec, overwrite=True)
    assert first.status == "success"

    first_run = pit.list_pipeline_runs(first.pipeline_id, limit=1)
    assert not first_run.empty
    first_anchor = pd.Timestamp(first_run.iloc[0]["max_output_asof"])
    assert first_anchor.tzinfo is not None

    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDP", "GDP"],
                "obs_date": [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-04-30")],
                "asof_utc": [
                    pd.Timestamp("2024-05-20", tz="UTC"),
                    pd.Timestamp("2024-05-20", tz="UTC"),
                ],
                "value": [3.1, 4.0],
            }
        )
    )

    explain = pit.explain_pipeline(spec, incremental=True)
    assert pd.Timestamp(explain["effective_start_asof"]) == first_anchor

    second = pit.apply_pipeline(spec, incremental=True)
    assert second.status == "success"
    assert second.incremental is True
    assert second.effective_start_asof == first_anchor
    assert second.rows_written > 0

    runs = pit.list_pipeline_runs(first.pipeline_id, limit=2)
    assert runs.shape[0] == 2
    assert bool(runs.iloc[0]["incremental"]) is True
    assert pd.Timestamp(runs.iloc[0]["effective_start_asof"]) == first_anchor


def test_pipeline_since_run_id_validation(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    with pytest.raises(PITContractError, match="incremental=True"):
        pit.preview_pipeline(_sample_pipeline(), since_asof=pd.Timestamp("2024-04-01", tz="UTC"))

    with pytest.raises(PITContractError, match="Unknown PIT pipeline run_id"):
        pit.apply_pipeline(_sample_pipeline(), incremental=True, since_run_id="does-not-exist")


def test_pipeline_outputs_available_through_pit_source(tmp_path):
    store = DuckDBParquetStore(root=str(tmp_path))
    pit = PITAccessor(store.conn())
    pit.upsert_pit_observations(_sample_df())
    pit.apply_pipeline(_sample_pipeline(), overwrite=True)

    source = PITDataSource(pit=pit)
    ctx = DataContext(sources={"pit": source}, calendars={}, store=store)

    panel = ctx.fetch_panel(
        "pit",
        Query(
            table="pit.snapshot",
            columns=["value", "asof_utc"],
            entities=["GDP_lag1_diff"],
            asof=pd.Timestamp("2024-05-30", tz="UTC"),
        ),
    )

    assert not panel.df.empty
    assert set(panel.df.index.get_level_values("entity_id")) == {"GDP_lag1_diff"}
