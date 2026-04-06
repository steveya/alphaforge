import pandas as pd

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.models import PITExpressionGraphSpec, PITExpressionNode
from alphaforge.pit.transforms import PITTransformSpec
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


def test_get_series_lineage_and_summary_for_transform_output(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    spec = PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_MINUS_CPI",
        op="binary",
        params={"right_series_key": "CPI", "operator": "sub", "join": "inner"},
        engine="python",
    )
    pit.apply_transform(spec, overwrite=True)

    lineage = pit.get_series_lineage("GDP_MINUS_CPI")

    assert not lineage.empty
    assert {"transform_id", "input_series_keys", "max_source_asof_utc", "causality_status"}.issubset(
        lineage.columns
    )
    assert lineage["lineage_kind"].eq("transform").all()
    assert lineage["causality_status"].eq("ok").all()
    assert all(keys == ("GDP", "CPI") for keys in lineage["input_series_keys"])
    assert (lineage["max_source_asof_utc"] <= lineage["asof_utc"]).all()

    summary = pit.explain_series("GDP_MINUS_CPI")
    assert summary["derived_row_count"] == len(lineage)
    assert summary["input_series_keys"] == ["CPI", "GDP"]
    assert summary["transform_ids"] == [spec.transform_id()]
    assert summary["causality_safe"] is True


def test_get_series_lineage_and_summary_for_expression_graph_output(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    graph = PITExpressionGraphSpec(
        graph_id="macro/gdp_expr",
        nodes=(
            PITExpressionNode(
                name="spread",
                output_series_key="GDP_MINUS_CPI_EXPR",
                expression="gdp - cpi",
                inputs={"gdp": "GDP", "cpi": "CPI"},
                join="inner",
            ),
        ),
    )
    pit.apply_expression_graph(graph, overwrite=True)

    lineage = pit.get_series_lineage("GDP_MINUS_CPI_EXPR")

    assert not lineage.empty
    assert lineage["lineage_kind"].eq("expression_graph").all()
    assert lineage["graph_id"].eq("macro/gdp_expr").all()
    assert lineage["node_name"].eq("spread").all()
    assert lineage["causality_status"].eq("ok").all()

    summary = pit.explain_series("GDP_MINUS_CPI_EXPR")
    assert summary["graph_ids"] == ["macro/gdp_expr"]
    assert summary["causality_safe"] is True


def test_explain_series_handles_raw_series_without_lineage(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_cross_df())

    summary = pit.explain_series("GDP")

    assert summary["derived_row_count"] == 0
    assert summary["lineage_kinds"] == ["raw"]
    assert summary["causality_status_counts"] == {"raw": 2}
