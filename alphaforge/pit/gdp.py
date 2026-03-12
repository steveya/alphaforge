from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd

from .accessor import _PIT_TABLE, PITAccessor, to_utc_aware, to_utc_naive
from .models import PITExpressionGraphResult, PITExpressionGraphSpec, PITExpressionNode

GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY = "GDPC1_QOQ_SAAR_RF_RT"
GDPC1_QOQ_SAAR_RF_REL_SERIES_KEY = "GDPC1_QOQ_SAAR_RF_REL"
GDPC1_QOQ_SAAR_RF_RT_GRAPH_ID = "macro/gdpc1_qoq_saar_rf_rt"
GDPC1_QOQ_SAAR_RF_REL_GRAPH_ID = "macro/gdpc1_qoq_saar_rf_rel"
QOQ_SAAR_FORMULA = "100 * ((L_t / L_{t-1})^4 - 1)"


def _qoq_saar_from_levels(level_t: float | None, level_tm1: float | None) -> float | None:
    if level_t is None or level_tm1 is None:
        return None
    if not pd.notna(level_t) or not pd.notna(level_tm1):
        return None
    left = float(level_t)
    right = float(level_tm1)
    if left <= 0.0 or right <= 0.0:
        return None
    return float(100.0 * ((left / right) ** 4.0 - 1.0))


def gdpc1_qoq_saar_rf_rt_graph(
    *,
    level_series_key: str = "GDPC1",
    output_series_key: str = GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY,
    graph_id: str = GDPC1_QOQ_SAAR_RF_RT_GRAPH_ID,
) -> PITExpressionGraphSpec:
    ratio = "(g / lag(g, 1))"
    power4 = f"(({ratio}) * ({ratio}) * ({ratio}) * ({ratio}))"
    expression = f"100 * ({power4} - 1)"
    return PITExpressionGraphSpec(
        graph_id=graph_id,
        description=(
            "Realtime same-asof rebase-free GDP QoQ SAAR derived from latest level pairs."
        ),
        nodes=(
            PITExpressionNode(
                name="growth_rf_rt",
                output_series_key=output_series_key,
                expression=expression,
                inputs={"g": level_series_key},
                join="inner",
            ),
        ),
    )


def gdpc1_qoq_saar_rf_rel_graph(
    *,
    level_series_key: str = "GDPC1",
    output_series_key: str = GDPC1_QOQ_SAAR_RF_REL_SERIES_KEY,
    graph_id: str = GDPC1_QOQ_SAAR_RF_REL_GRAPH_ID,
) -> PITExpressionGraphSpec:
    return PITExpressionGraphSpec(
        graph_id=graph_id,
        description=(
            "Release-anchored same-asof rebase-free GDP QoQ SAAR derived from GDP release events."
        ),
        nodes=(
            PITExpressionNode(
                name="growth_rf_rel",
                output_series_key=output_series_key,
                expression="g",
                inputs={"g": level_series_key},
                join="inner",
            ),
        ),
    )


def apply_gdpc1_qoq_saar_rf_rt(
    pit: PITAccessor,
    *,
    level_series_key: str = "GDPC1",
    output_series_key: str = GDPC1_QOQ_SAAR_RF_RT_SERIES_KEY,
    graph_id: str = GDPC1_QOQ_SAAR_RF_RT_GRAPH_ID,
    start_obs: pd.Timestamp | None = None,
    end_obs: pd.Timestamp | None = None,
    start_asof: pd.Timestamp | None = None,
    end_asof: pd.Timestamp | None = None,
    overwrite: bool = False,
    incremental: bool = False,
    since_asof: pd.Timestamp | None = None,
    since_run_id: str | None = None,
) -> PITExpressionGraphResult:
    return pit.apply_expression_graph(
        gdpc1_qoq_saar_rf_rt_graph(
            level_series_key=level_series_key,
            output_series_key=output_series_key,
            graph_id=graph_id,
        ),
        start_obs=start_obs,
        end_obs=end_obs,
        start_asof=start_asof,
        end_asof=end_asof,
        overwrite=overwrite,
        incremental=incremental,
        since_asof=since_asof,
        since_run_id=since_run_id,
    )


def _build_release_anchored_growth_rows(
    pit: PITAccessor,
    *,
    level_series_key: str,
    output_series_key: str,
    graph_id: str,
    start_obs: pd.Timestamp | None,
    end_obs: pd.Timestamp | None,
    start_asof: pd.Timestamp | None,
    end_asof: pd.Timestamp | None,
) -> pd.DataFrame:
    obs_filters: list[str] = ["series_key = ?"]
    obs_params: list[object] = [level_series_key]
    if start_obs is not None:
        obs_filters.append("obs_date >= ?")
        obs_params.append(to_utc_naive(start_obs))
    if end_obs is not None:
        obs_filters.append("obs_date <= ?")
        obs_params.append(to_utc_naive(end_obs))
    obs_query = f"""
        SELECT DISTINCT obs_date
        FROM {_PIT_TABLE}
        WHERE {' AND '.join(obs_filters)}
        ORDER BY obs_date ASC
    """
    obs_rows = pit.conn.execute(obs_query, obs_params).fetchdf()
    if obs_rows.empty:
        return pd.DataFrame(
            columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
        )

    outputs: list[dict[str, Any]] = []
    source_name = f"pit_expr_graph:{graph_id}:growth_rf_rel"
    for obs_value in obs_rows["obs_date"].tolist():
        obs_date = to_utc_aware(obs_value)
        timeline = pit.get_revision_path(
            level_series_key,
            obs_date,
            start_asof=start_asof,
            end_asof=end_asof,
        )
        if timeline.empty:
            continue
        for row in timeline.itertuples(index=False):
            asof_utc = pd.Timestamp(row.asof_utc)
            snapshot = pit.get_snapshot(level_series_key, asof_utc, end=obs_date)
            if snapshot.empty or obs_date not in snapshot.index:
                continue
            current_pos = snapshot.index.get_loc(obs_date)
            if isinstance(current_pos, slice):
                current_pos = current_pos.stop - 1
            if int(current_pos) <= 0:
                continue
            prev_obs_date = pd.Timestamp(snapshot.index[int(current_pos) - 1])
            value = _qoq_saar_from_levels(
                float(snapshot.iloc[int(current_pos)]),
                float(snapshot.iloc[int(current_pos) - 1]),
            )
            if value is None:
                continue
            lineage = json.dumps(
                {
                    "graph_id": graph_id,
                    "node_name": "growth_rf_rel",
                    "output_series_key": output_series_key,
                    "formula": QOQ_SAAR_FORMULA,
                    "lineage_mode": "release_anchored_same_asof_pair",
                    "inputs": {"g": level_series_key},
                    "source_asof_by_series_utc": {level_series_key: asof_utc.isoformat()},
                    "source_obs_dates_utc": [
                        prev_obs_date.isoformat(),
                        obs_date.isoformat(),
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            outputs.append(
                {
                    "series_key": output_series_key,
                    "obs_date": obs_date,
                    "asof_utc": asof_utc,
                    "value": value,
                    "source": source_name,
                    "meta_json": lineage,
                }
            )

    if not outputs:
        return pd.DataFrame(
            columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
        )
    return pd.DataFrame(outputs).sort_values(["obs_date", "asof_utc"]).reset_index(drop=True)


def apply_gdpc1_qoq_saar_rf_rel(
    pit: PITAccessor,
    *,
    level_series_key: str = "GDPC1",
    output_series_key: str = GDPC1_QOQ_SAAR_RF_REL_SERIES_KEY,
    graph_id: str = GDPC1_QOQ_SAAR_RF_REL_GRAPH_ID,
    start_obs: pd.Timestamp | None = None,
    end_obs: pd.Timestamp | None = None,
    start_asof: pd.Timestamp | None = None,
    end_asof: pd.Timestamp | None = None,
    overwrite: bool = False,
    incremental: bool = False,
    since_asof: pd.Timestamp | None = None,
    since_run_id: str | None = None,
) -> PITExpressionGraphResult:
    started_utc = pd.Timestamp.now(tz="UTC")
    run_id = str(uuid.uuid4())
    spec = gdpc1_qoq_saar_rf_rel_graph(
        level_series_key=level_series_key,
        output_series_key=output_series_key,
        graph_id=graph_id,
    )
    resolved_graph_id = pit._upsert_expression_graph_metadata(spec)
    effective_start_asof = pit._resolve_expression_graph_effective_start_asof(
        graph_id=resolved_graph_id,
        incremental=incremental,
        start_asof=start_asof,
        since_asof=since_asof,
        since_run_id=since_run_id,
    )
    rows_written = 0
    max_output_asof: pd.Timestamp | None = None
    try:
        result_df = _build_release_anchored_growth_rows(
            pit,
            level_series_key=level_series_key,
            output_series_key=output_series_key,
            graph_id=resolved_graph_id,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=effective_start_asof,
            end_asof=end_asof,
        )
        if overwrite:
            pit._delete_transformed_rows(
                output_series_key=output_series_key,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=effective_start_asof,
                end_asof=end_asof,
            )
        if not result_df.empty:
            pit.upsert_pit_observations(result_df, strict=False)
            rows_written = int(len(result_df))
            max_output_asof = pd.Timestamp(result_df["asof_utc"].max())
        finished_utc = pd.Timestamp.now(tz="UTC")
        pit._insert_expression_graph_run(
            run_id=run_id,
            graph_id=resolved_graph_id,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
            incremental=incremental,
            requested_since_asof=since_asof,
            effective_start_asof=effective_start_asof,
            requested_since_run_id=since_run_id,
            max_output_asof=max_output_asof,
            rows_written=rows_written,
            node_count=1,
            status="success",
            started_utc=started_utc,
            finished_utc=finished_utc,
        )
    except Exception:
        finished_utc = pd.Timestamp.now(tz="UTC")
        pit._insert_expression_graph_run(
            run_id=run_id,
            graph_id=resolved_graph_id,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
            incremental=incremental,
            requested_since_asof=since_asof,
            effective_start_asof=effective_start_asof,
            requested_since_run_id=since_run_id,
            max_output_asof=max_output_asof,
            rows_written=rows_written,
            node_count=1,
            status="failed",
            started_utc=started_utc,
            finished_utc=finished_utc,
        )
        raise

    return PITExpressionGraphResult(
        graph_id=resolved_graph_id,
        run_id=run_id,
        status="success",
        rows_written=rows_written,
        node_rows_written={"growth_rf_rel": rows_written},
        run_started_utc=started_utc,
        run_finished_utc=finished_utc,
        incremental=incremental,
        effective_start_asof=effective_start_asof,
    )
