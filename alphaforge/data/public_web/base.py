from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema

from .finalize import empty_frame_for_schema, finalize_public_frame, frame_from_records
from .http import CachedHttpClient


class PublicWebSourceBase:
    name: str

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        now_fn: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._now_fn = now_fn or (lambda: pd.Timestamp.now(tz="UTC"))

    def _default_table(self) -> str:
        table = getattr(self, "TABLE", None)
        if not isinstance(table, str) or not table:
            raise AttributeError(f"{type(self).__name__} must define TABLE")
        return table

    def _schema(self, table: str | None = None) -> TableSchema:
        resolved_table = table or self._default_table()
        return self.schemas()[resolved_table]

    def _require_table(self, q: Query, expected: str | None = None) -> str:
        resolved_expected = expected or self._default_table()
        if q.table != resolved_expected:
            raise ValueError(f"Unknown table: {q.table}")
        return resolved_expected

    def _require_entities(self, q: Query, *, error_message: str) -> list[str]:
        entities = [str(entity) for entity in (q.entities or [])]
        if not entities:
            raise ValueError(error_message)
        return entities

    def _now_utc(self) -> pd.Timestamp:
        now = pd.Timestamp(self._now_fn())
        if now.tzinfo is None:
            return now.tz_localize("UTC")
        return now.tz_convert("UTC")

    def _asof_utc(self, q: Query) -> pd.Timestamp:
        return q.asof or self._now_utc()

    def _empty_frame(
        self,
        schema: TableSchema,
        *,
        time_col: str | None = None,
        entity_col: str | None = None,
    ) -> pd.DataFrame:
        return empty_frame_for_schema(schema, time_col=time_col, entity_col=entity_col)

    def _frame_from_records(
        self,
        records: Sequence[dict] | Iterable[dict],
        *,
        schema: TableSchema,
        time_col: str | None = None,
        entity_col: str | None = None,
    ) -> pd.DataFrame:
        return frame_from_records(
            records,
            schema=schema,
            time_col=time_col,
            entity_col=entity_col,
        )

    def _finalize(
        self,
        df: pd.DataFrame,
        *,
        q: Query,
        schema: TableSchema,
        time_col: str | None = None,
        entity_col: str | None = None,
        sort_by: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return finalize_public_frame(
            df,
            q=q,
            schema=schema,
            time_col=time_col,
            entity_col=entity_col,
            sort_by=sort_by,
        )
