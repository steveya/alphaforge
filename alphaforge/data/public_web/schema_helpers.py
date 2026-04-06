from __future__ import annotations

from collections.abc import Sequence

from alphaforge.data.schema import TableSchema


def _normalize_columns(columns: Sequence[str] | None) -> list[str]:
    return [str(column) for column in (columns or [])]


def table_schema(
    name: str,
    *,
    required_columns: Sequence[str],
    canonical_columns: Sequence[str] | None = None,
    entity_column: str = "entity_id",
    time_column: str = "date",
    native_freq: str | None = None,
    time_semantics: str | None = None,
    expected_cadence_days: int | None = None,
    event_time_column: str | None = None,
    release_time_column: str | None = None,
    revision_id_column: str | None = None,
) -> TableSchema:
    canonical = _normalize_columns(canonical_columns) or _normalize_columns(
        required_columns
    )
    return TableSchema(
        name=name,
        required_columns=_normalize_columns(required_columns),
        canonical_columns=canonical,
        entity_column=entity_column,
        time_column=time_column,
        native_freq=native_freq,
        time_semantics=time_semantics,
        expected_cadence_days=expected_cadence_days,
        event_time_column=event_time_column,
        release_time_column=release_time_column,
        revision_id_column=revision_id_column,
    )


def single_value_schema(
    name: str,
    *,
    value_column: str = "value",
    entity_column: str = "entity_id",
    time_column: str = "date",
    native_freq: str | None = None,
    time_semantics: str | None = None,
    expected_cadence_days: int | None = None,
) -> TableSchema:
    return table_schema(
        name,
        required_columns=[value_column],
        canonical_columns=[value_column],
        entity_column=entity_column,
        time_column=time_column,
        native_freq=native_freq,
        time_semantics=time_semantics,
        expected_cadence_days=expected_cadence_days,
    )


def daily_panel_schema(
    name: str,
    *,
    required_columns: Sequence[str],
    canonical_columns: Sequence[str] | None = None,
    entity_column: str = "entity_id",
    time_column: str = "date",
) -> TableSchema:
    return table_schema(
        name,
        required_columns=required_columns,
        canonical_columns=canonical_columns,
        entity_column=entity_column,
        time_column=time_column,
        native_freq="D",
        expected_cadence_days=1,
    )


def event_table_schema(
    name: str,
    *,
    required_columns: Sequence[str],
    canonical_columns: Sequence[str] | None = None,
    entity_column: str = "entity_id",
    time_column: str = "ts_utc",
    native_freq: str | None = None,
    time_semantics: str | None = None,
    expected_cadence_days: int | None = None,
    release_time_column: str | None = "asof_utc",
    revision_id_column: str | None = None,
) -> TableSchema:
    return table_schema(
        name,
        required_columns=required_columns,
        canonical_columns=canonical_columns,
        entity_column=entity_column,
        time_column=time_column,
        native_freq=native_freq,
        time_semantics=time_semantics,
        expected_cadence_days=expected_cadence_days,
        event_time_column=time_column,
        release_time_column=release_time_column,
        revision_id_column=revision_id_column,
    )
