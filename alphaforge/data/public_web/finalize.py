from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema

from .utils import apply_query_filters, project_columns


def _ordered_unique(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            seen.add(column)
            out.append(column)
    return out


def schema_frame_columns(
    schema: TableSchema,
    *,
    time_col: str | None = None,
    entity_col: str | None = None,
) -> list[str]:
    return _ordered_unique(
        [
            time_col or schema.time_column,
            entity_col or schema.entity_column,
            "asof_utc",
            *schema.required_columns,
            *schema.canonical_columns,
        ]
    )


def empty_frame_for_schema(
    schema: TableSchema,
    *,
    time_col: str | None = None,
    entity_col: str | None = None,
) -> pd.DataFrame:
    columns = schema_frame_columns(schema, time_col=time_col, entity_col=entity_col)
    return pd.DataFrame(columns=columns)


def frame_from_records(
    records: Sequence[dict] | Iterable[dict],
    *,
    schema: TableSchema,
    time_col: str | None = None,
    entity_col: str | None = None,
) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return empty_frame_for_schema(schema, time_col=time_col, entity_col=entity_col)
    return pd.DataFrame.from_records(rows)


def finalize_public_frame(
    df: pd.DataFrame,
    *,
    q: Query,
    schema: TableSchema,
    time_col: str | None = None,
    entity_col: str | None = None,
    sort_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    resolved_time_col = time_col or schema.time_column
    resolved_entity_col = entity_col or schema.entity_column

    if df.empty:
        return empty_frame_for_schema(
            schema,
            time_col=resolved_time_col,
            entity_col=resolved_entity_col,
        )

    out = apply_query_filters(
        df,
        q=q,
        time_col=resolved_time_col,
        entity_col=resolved_entity_col,
    )
    out = project_columns(
        out,
        required_columns=schema.required_columns,
        requested_columns=q.columns,
        time_col=resolved_time_col,
        entity_col=resolved_entity_col,
    )
    sort_columns = [
        column
        for column in (
            list(sort_by)
            if sort_by is not None
            else [resolved_entity_col, resolved_time_col]
        )
        if column in out.columns
    ]
    if sort_columns:
        out = out.sort_values(sort_columns)
    return out.reset_index(drop=True)
