from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .parsing import parse_csv_bytes, parse_html_tables, parse_xlsx_bytes
from .utils import ensure_date_utc, first_existing, to_float


def artifact_name_from_url(url: str, fallback: str) -> str:
    path = Path(str(url).split("?", 1)[0])
    return path.name or fallback


def candidate_tables(
    tables: Iterable[pd.DataFrame],
    *,
    any_of: Iterable[str] | None = None,
    all_of: Iterable[str] | None = None,
) -> list[pd.DataFrame]:
    any_columns = {str(column) for column in (any_of or [])}
    all_columns = {str(column) for column in (all_of or [])}

    selected: list[pd.DataFrame] = []
    for table in tables:
        columns = set(table.columns)
        if any_columns and not (columns & any_columns):
            continue
        if all_columns and not all_columns.issubset(columns):
            continue
        selected.append(table)
    return selected


def resolved_date_series(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    default_date: pd.Timestamp,
) -> pd.Series:
    column = first_existing(frame, *aliases)
    if column is None:
        return pd.Series(default_date, index=frame.index)
    return ensure_date_utc(frame[column])


def resolved_numeric_series(
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> pd.Series:
    column = first_existing(frame, *aliases)
    if column is None:
        return pd.Series(pd.NA, index=frame.index)
    return to_float(frame[column])


def resolved_text_series(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    default: str,
    case: str | None = None,
    space_replacement: str | None = None,
) -> pd.Series:
    column = first_existing(frame, *aliases)
    if column is None:
        series = pd.Series(default, index=frame.index)
    else:
        raw = frame[column].where(frame[column].notna(), default)
        series = raw.astype(str).str.strip()

    if case == "lower":
        series = series.str.lower()
    elif case == "upper":
        series = series.str.upper()

    if space_replacement is not None:
        series = series.str.replace(" ", space_replacement, regex=False)

    return series


class TabularDocumentSourceBase(PublicWebSourceBase):
    def _artifact_name_from_url(self, url: str, fallback: str) -> str:
        return artifact_name_from_url(url, fallback)

    def _read_html_tables(
        self,
        *,
        url: str,
        source: str,
        artifact_name: str,
    ) -> list[pd.DataFrame]:
        payload = self._http.get_bytes(
            url=url,
            source=source,
            artifact_name=artifact_name,
        )
        return parse_html_tables(payload)

    def _read_xlsx_frame(
        self,
        *,
        url: str,
        source: str,
        artifact_name: str,
    ) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=url,
            source=source,
            artifact_name=artifact_name,
        )
        return parse_xlsx_bytes(payload)

    def _read_csv_frame(
        self,
        *,
        url: str,
        source: str,
        artifact_name: str,
    ) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=url,
            source=source,
            artifact_name=artifact_name,
        )
        return parse_csv_bytes(payload)

    def _snapshot_date(self, q: Query) -> pd.Timestamp:
        return self._asof_utc(q).normalize()
