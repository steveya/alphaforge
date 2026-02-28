from __future__ import annotations

import io
import zipfile
from html.parser import HTMLParser
from typing import Iterable

import pandas as pd

from .utils import snake_case


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake_case(str(c)) for c in out.columns]
    return out


def parse_csv_bytes(data: bytes, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), **kwargs)
    return normalize_headers(df)


def parse_zip_csv_bytes(data: bytes, **kwargs) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        for member in members:
            with zf.open(member) as handle:
                frame = pd.read_csv(handle, **kwargs)
            frame = normalize_headers(frame)
            frame["_source_file"] = member
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_xlsx_bytes(
    data: bytes, *, sheet_names: Iterable[str] | None = None
) -> pd.DataFrame:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    frames: list[pd.DataFrame] = []
    for name, frame in sheets.items():
        if sheet_names is not None and name not in set(sheet_names):
            continue
        out = normalize_headers(frame)
        out["_sheet_name"] = name
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_html_tables(html: str | bytes | bytearray | memoryview) -> list[pd.DataFrame]:
    html_text = html if isinstance(html, str) else bytes(html).decode()
    try:
        tables = pd.read_html(io.StringIO(html_text))
        return [normalize_headers(frame) for frame in tables]
    except Exception:
        parser = _SimpleHTMLTableParser()
        parser.feed(html_text)
        return [normalize_headers(frame) for frame in parser.to_frames()]


class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._in_header = False
        self._current_cell: list[str] = []
        self._current_row: list[str] = []
        self._headers: list[str] | None = None
        self._rows: list[list[str]] = []
        self._tables: list[tuple[list[str] | None, list[list[str]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._headers = None
            self._rows = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_table and self._in_row and tag in {"th", "td"}:
            self._in_cell = True
            self._in_header = tag == "th"
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_table and self._in_row and tag in {"th", "td"} and self._in_cell:
            value = "".join(self._current_cell).strip()
            self._current_row.append(value)
            self._in_cell = False
            self._in_header = False
        elif self._in_table and tag == "tr" and self._in_row:
            if self._current_row:
                if self._headers is None and self._inferred_header_row(
                    self._current_row
                ):
                    self._headers = self._current_row
                else:
                    self._rows.append(self._current_row)
            self._in_row = False
        elif tag == "table" and self._in_table:
            self._tables.append((self._headers, self._rows))
            self._in_table = False

    @staticmethod
    def _inferred_header_row(row: list[str]) -> bool:
        if not row:
            return False
        non_numeric = sum(
            1 for x in row if not x.replace(",", "").replace(".", "").isdigit()
        )
        return non_numeric >= max(1, len(row) // 2)

    def to_frames(self) -> list[pd.DataFrame]:
        out: list[pd.DataFrame] = []
        for headers, rows in self._tables:
            if not rows:
                continue
            if headers and len(headers) == len(rows[0]):
                out.append(pd.DataFrame(rows, columns=headers))
            else:
                out.append(pd.DataFrame(rows))
        return out
