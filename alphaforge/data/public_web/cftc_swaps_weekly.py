from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .parsing import parse_csv_bytes, parse_xlsx_bytes
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    first_existing,
    make_entity_id,
    project_columns,
    to_float,
)


class CFTCWeeklySwapsSource(DataSource):
    name: str = "cftc_swaps_weekly"
    TABLE = "cftc.swaps.weekly"
    ARCHIVE_URL = "https://www.cftc.gov/MarketReports/SwapsReports/Archive/index.htm"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        archive_url: str | None = None,
        file_urls: list[str] | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._archive_url = archive_url or self.ARCHIVE_URL
        self._file_urls = file_urls

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=[
                    "value",
                    "report_name",
                    "metric",
                    "currency",
                    "maturity_bucket",
                    "participant_type",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq="W",
                time_semantics="interval_end",
            )
        }

    def _discover_file_urls(self) -> list[str]:
        if self._file_urls:
            return self._file_urls

        payload = self._http.get_bytes(
            url=self._archive_url,
            source="cftc_swaps_weekly",
            artifact_name="archive_index.html",
        )
        html = payload.decode(errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        files: list[str] = []
        for href in hrefs:
            if href.lower().endswith((".xlsx", ".xls", ".csv")):
                files.append(urljoin(self._archive_url, href))
        return sorted(set(files))

    def _read_file(self, url: str) -> pd.DataFrame:
        ext = url.lower().split("?")[0]
        payload = self._http.get_bytes(
            url=url,
            source="cftc_swaps_weekly",
            artifact_name=Path(ext).name or "weekly_file",
        )
        if ext.endswith(".csv"):
            return parse_csv_bytes(payload)
        return parse_xlsx_bytes(payload)

    def _to_long(self, frame: pd.DataFrame, report_name: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        out = pd.DataFrame(index=frame.index)
        date_col = first_existing(frame, "date", "week_ending", "report_date")
        out["date"] = ensure_date_utc(frame[date_col]) if date_col else pd.NaT
        out["report_name"] = report_name

        out["metric"] = (
            frame[first_existing(frame, "metric")].astype(str)
            if first_existing(frame, "metric")
            else "value"
        )
        out["currency"] = (
            frame[first_existing(frame, "currency", "ccy")].astype(str).str.lower()
            if first_existing(frame, "currency", "ccy")
            else "all"
        )
        out["maturity_bucket"] = (
            frame[first_existing(frame, "maturity_bucket", "bucket", "tenor")]
            .astype(str)
            .str.lower()
            if first_existing(frame, "maturity_bucket", "bucket", "tenor")
            else "all"
        )
        out["participant_type"] = (
            frame[first_existing(frame, "participant_type", "participant")]
            .astype(str)
            .str.lower()
            .str.replace(" ", "_", regex=False)
            if first_existing(frame, "participant_type", "participant")
            else "all"
        )

        value_col = first_existing(
            frame, "value", "notional", "amount", "trade_count", "count"
        )
        if value_col is None:
            numeric_cols = [
                c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
            ]
            value_col = numeric_cols[0] if numeric_cols else frame.columns[-1]
        out["value"] = to_float(frame[value_col])

        out["entity_id"] = [
            make_entity_id(
                "rates",
                "irs",
                ccy,
                bucket,
                metric,
                "cftc",
                participant,
            )
            for ccy, bucket, metric, participant in zip(
                out["currency"],
                out["maturity_bucket"],
                out["metric"],
                out["participant_type"],
            )
        ]

        out["asof_utc"] = out["date"]
        return out

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        frames: list[pd.DataFrame] = []
        for url in self._discover_file_urls():
            parsed = self._read_file(url)
            long_df = self._to_long(parsed, Path(url).name)
            if not long_df.empty:
                frames.append(long_df)

        schema = self.schemas()[self.TABLE]
        if not frames:
            return pd.DataFrame(
                columns=[
                    schema.time_column,
                    schema.entity_column,
                    "asof_utc",
                    *schema.required_columns,
                ]
            )

        out = pd.concat(frames, ignore_index=True)
        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
