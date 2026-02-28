import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError, PITValidationError
from alphaforge.pit.models import PITExpressionGraphSpec, PITExpressionNode
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_cross_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "CPI", "CPI"],
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
            ],
            "asof_utc": [
                pd.Timestamp("2024-03-10", tz="UTC"),
                pd.Timestamp("2024-03-10", tz="UTC"),
                pd.Timestamp("2024-03-05", tz="UTC"),
                pd.Timestamp("2024-03-05", tz="UTC"),
            ],
            "value": [3.0, 3.5, 1.0, 1.1],
        }
    )


def _graph() -> PITExpressionGraphSpec:
    return PITExpressionGraphSpec(
        graph_id="macro/gdp_expr",
        nodes=(
            PITExpressionNode(
                name="spread",
                output_series_key="GDP_minus_CPI_expr",
                expression="gdp - cpi",
                inputs={"gdp": "GDP", "cpi": "CPI"},
                join="inner",
            ),
        ),
    )


def test_expression_graph_preview_apply_and_lineage(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    explain = pit.explain_expression_graph(_graph())
    assert explain["node_count"] == 1
    assert explain["nodes"][0]["candidate_asof_count"] > 0

    preview = pit.preview_expression_graph(_graph(), overwrite=True)
    assert not preview.empty

    result = pit.apply_expression_graph(_graph(), overwrite=True)
    assert result.status == "success"
    assert result.rows_written == len(preview)

    got = pit.conn.execute(
        """
        SELECT obs_date, asof_utc, value, meta_json
        FROM pit_observations
        WHERE series_key = 'GDP_minus_CPI_expr'
        ORDER BY obs_date, asof_utc
        """
    ).fetchdf()
    assert len(got) == len(preview)
    assert "source_asof_by_series_utc" in str(got.iloc[0]["meta_json"])


def test_expression_graph_supports_lag_and_diff(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDP", "GDP", "GDP"],
                "obs_date": [
                    pd.Timestamp("2024-01-31"),
                    pd.Timestamp("2024-02-29"),
                    pd.Timestamp("2024-03-31"),
                ],
                "asof_utc": [
                    pd.Timestamp("2024-04-01", tz="UTC"),
                    pd.Timestamp("2024-04-01", tz="UTC"),
                    pd.Timestamp("2024-04-01", tz="UTC"),
                ],
                "value": [1.0, 2.0, 4.0],
            }
        )
    )

    graph = PITExpressionGraphSpec(
        graph_id="macro/gdp_ops",
        nodes=(
            PITExpressionNode(
                name="expr",
                output_series_key="GDP_expr_ops",
                expression="diff(g, 1) + lag(g, 1)",
                inputs={"g": "GDP"},
            ),
        ),
    )
    preview = pit.preview_expression_graph(graph)
    assert not preview.empty


def test_expression_graph_validates_aliases_and_dependencies(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    bad_alias = PITExpressionGraphSpec(
        nodes=(
            PITExpressionNode(
                name="x",
                output_series_key="BAD_EXPR",
                expression="foo + 1",
                inputs={"g": "GDP"},
            ),
        )
    )
    with pytest.raises(PITValidationError, match="Unknown expression alias"):
        pit.preview_expression_graph(bad_alias)

    bad_dep = PITExpressionGraphSpec(
        nodes=(
            PITExpressionNode(
                name="x",
                output_series_key="BAD_DEP",
                expression="g",
                inputs={"g": "GDP"},
                depends_on=("missing",),
            ),
        )
    )
    with pytest.raises(PITContractError, match="depends on unknown"):
        pit.explain_expression_graph(bad_dep)


def test_expression_graph_incremental_anchor(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    first = pit.apply_expression_graph(_graph(), overwrite=True)
    assert first.status == "success"

    pit.upsert_pit_observations(
        pd.DataFrame(
            {
                "series_key": ["GDP", "CPI"],
                "obs_date": [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-03-31")],
                "asof_utc": [
                    pd.Timestamp("2024-04-20", tz="UTC"),
                    pd.Timestamp("2024-04-20", tz="UTC"),
                ],
                "value": [4.0, 1.2],
            }
        )
    )

    second = pit.apply_expression_graph(_graph(), incremental=True)
    assert second.incremental is True
    assert second.effective_start_asof is not None
