from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .archive import (
    discover_archive_fetches,
    plan_archive_fetches,
    read_first_zip_member,
)
from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .parsing import parse_csv_bytes, parse_xlsx_bytes
from .schema_helpers import table_schema
from .utils import (
    ensure_date_utc,
    first_existing,
    make_entity_id,
    to_float,
)


class CFTCWeeklySwapsSource(PublicWebSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._archive_url = archive_url or self.ARCHIVE_URL
        self._file_urls = file_urls

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["value"],
                canonical_columns=[
                    "value",
                    "report_name",
                    "metric",
                    "currency",
                    "maturity_bucket",
                    "participant_type",
                ],
                native_freq="W",
                time_semantics="interval_end",
            )
        }

    def _discover_file_urls(self, q: Query):
        if self._file_urls:
            return plan_archive_fetches(
                self._file_urls,
                years=None,
                fallback_artifact_prefix="cftc_swaps_weekly",
            )

        payload = self._http.get_bytes(
            url=self._archive_url,
            source="cftc_swaps_weekly",
            artifact_name="archive_index.html",
        )
        html = payload.decode(errors="ignore")
        years: set[int] = set()
        if q.start is not None:
            years.add(q.start.year)
        if q.end is not None:
            years.add(q.end.year)
        return discover_archive_fetches(
            html,
            base_url=self._archive_url,
            suffixes=(".xlsx", ".xls", ".csv", ".zip"),
            years=years,
            fallback_artifact_prefix="cftc_swaps_weekly",
        )

    def _read_file(self, planned) -> pd.DataFrame:
        ext = planned.url.lower().split("?", 1)[0]
        payload = self._http.get_bytes(
            url=planned.url,
            source="cftc_swaps_weekly",
            artifact_name=planned.artifact_name,
        )
        if ext.endswith(".zip"):
            member = read_first_zip_member(payload, suffixes=(".csv", ".xlsx", ".xls"))
            if member is None:
                return pd.DataFrame()
            member_name, member_payload = member
            member_ext = member_name.lower()
            if member_ext.endswith(".csv"):
                return parse_csv_bytes(member_payload)
            return parse_xlsx_bytes(member_payload)
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
        self._require_table(q)

        frames: list[pd.DataFrame] = []
        for planned in self._discover_file_urls(q):
            parsed = self._read_file(planned)
            long_df = self._to_long(parsed, planned.artifact_name)
            if not long_df.empty:
                frames.append(long_df)

        schema = self._schema()
        if not frames:
            return self._empty_frame(schema)

        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, q=q, schema=schema, sort_by=[])
