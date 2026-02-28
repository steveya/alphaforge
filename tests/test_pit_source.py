import pandas as pd
import pytest

from alphaforge.data.context import DataContext
from alphaforge.data.pit_source import PITDataSource
from alphaforge.data.query import Query
from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError, PITUnsupportedOperationError
from alphaforge.pit.guards import ReleaseLagPolicy, effective_asof
from alphaforge.pit.transforms import PITTransformSpec
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_ctx(tmp_path) -> tuple[DataContext, PITAccessor]:
    store = DuckDBParquetStore(root=str(tmp_path))
    pit = PITAccessor(store.conn())
    source = PITDataSource(pit=pit)
    ctx = DataContext(sources={"pit": source}, calendars={}, store=store)
    return ctx, pit


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-03-31"),
                pd.Timestamp("2025-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-04-10", tz="UTC"),
                pd.Timestamp("2025-05-10", tz="UTC"),
            ],
            "value": [1.0, 1.1, 2.0, 2.1],
            "source": ["test"] * 4,
        }
    )


def test_pit_snapshot_source_fetches_panel(tmp_path):
    ctx, pit = _make_ctx(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    panel = ctx.fetch_panel(
        "pit",
        Query(
            table="pit.snapshot",
            columns=["value", "asof_utc"],
            entities=["GDP"],
            start=pd.Timestamp("2024-12-31", tz="UTC"),
            end=pd.Timestamp("2025-03-31", tz="UTC"),
            asof=pd.Timestamp("2025-04-15", tz="UTC"),
        ),
    )

    assert not panel.df.empty
    assert panel.df.index.names == ["ts_utc", "entity_id"]
    assert set(panel.df.index.get_level_values("entity_id")) == {"GDP"}
    assert panel.df["asof_utc"].max() <= pd.Timestamp("2025-04-15", tz="UTC")


def test_pit_source_reads_transformed_series(tmp_path):
    ctx, pit = _make_ctx(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    pit.apply_transform(
        PITTransformSpec(
            input_series_key="GDP",
            output_series_key="GDP_lag1",
            op="lag",
            params={"periods": 1},
        ),
        overwrite=True,
    )

    panel = ctx.fetch_panel(
        "pit",
        Query(
            table="pit.snapshot",
            columns=["value", "asof_utc"],
            entities=["GDP_lag1"],
            asof=pd.Timestamp("2025-05-10", tz="UTC"),
        ),
    )
    assert not panel.df.empty
    assert set(panel.df.index.get_level_values("entity_id")) == {"GDP_lag1"}


def test_pit_observations_table_respects_asof(tmp_path):
    ctx, pit = _make_ctx(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    panel = ctx.fetch_panel(
        "pit",
        Query(
            table="pit.observations",
            columns=["value", "asof_utc"],
            entities=["GDP"],
            asof=pd.Timestamp("2025-02-11", tz="UTC"),
        ),
    )

    assert not panel.df.empty
    assert panel.df["asof_utc"].max() <= pd.Timestamp("2025-02-11", tz="UTC")


def test_snapshot_lag_policy_matches_effective_asof(tmp_path):
    store = DuckDBParquetStore(root=str(tmp_path))
    pit = PITAccessor(store.conn())
    pit.upsert_pit_observations(_sample_df())

    policy = ReleaseLagPolicy(default_lag=pd.Timedelta(days=30))
    source = PITDataSource(pit=pit, lag_policy=policy)
    ctx = DataContext(sources={"pit": source}, calendars={}, store=store)

    asof = pd.Timestamp("2025-04-15", tz="UTC")
    panel = ctx.fetch_panel(
        "pit",
        Query(
            table="pit.snapshot",
            columns=["value", "asof_utc"],
            entities=["GDP"],
            asof=asof,
        ),
    )
    assert not panel.df.empty

    expected = pit.get_snapshot("GDP", effective_asof(asof, "GDP", policy))
    got = panel.df["value"]
    got.index = pd.DatetimeIndex(got.index.get_level_values("ts_utc"))
    got = got.sort_index()
    pd.testing.assert_series_equal(got, expected.reindex(got.index), check_names=False)


def test_snapshot_requires_asof(tmp_path):
    ctx, pit = _make_ctx(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    with pytest.raises(PITContractError, match="requires Query.asof"):
        ctx.fetch_panel(
            "pit",
            Query(table="pit.snapshot", columns=["value"], entities=["GDP"]),
        )


def test_snapshot_and_observations_reject_unsupported_vintage_modes(tmp_path):
    ctx, pit = _make_ctx(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    with pytest.raises(PITUnsupportedOperationError, match="vintage='latest'"):
        ctx.fetch_panel(
            "pit",
            Query(
                table="pit.snapshot",
                columns=["value"],
                entities=["GDP"],
                asof=pd.Timestamp("2025-04-15", tz="UTC"),
                vintage="first",
            ),
        )

    with pytest.raises(PITUnsupportedOperationError, match="vintage_id"):
        ctx.fetch_panel(
            "pit",
            Query(
                table="pit.snapshot",
                columns=["value"],
                entities=["GDP"],
                asof=pd.Timestamp("2025-04-15", tz="UTC"),
                vintage_id="foo",
            ),
        )

    with pytest.raises(PITUnsupportedOperationError, match="vintage='latest'"):
        ctx.fetch_panel(
            "pit",
            Query(
                table="pit.observations",
                columns=["value"],
                entities=["GDP"],
                vintage="first",
            ),
        )


def test_pit_query_builder_helpers():
    snapshot_q = PITDataSource.snapshot_query(
        columns=["value", "asof_utc"],
        entities=["GDP"],
        asof=pd.Timestamp("2025-04-15", tz="UTC"),
    )
    assert snapshot_q.table == PITDataSource.SNAPSHOT_TABLE
    assert snapshot_q.vintage == "latest"
    assert snapshot_q.asof == pd.Timestamp("2025-04-15", tz="UTC")

    observations_q = PITDataSource.observations_query(
        columns=["value", "asof_utc"],
        entities=["GDP"],
        asof=pd.Timestamp("2025-04-15", tz="UTC"),
    )
    assert observations_q.table == PITDataSource.OBS_TABLE
    assert observations_q.vintage == "latest"
