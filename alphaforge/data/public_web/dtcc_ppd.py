from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .parsing import parse_zip_csv_bytes
from .schema_helpers import event_table_schema, table_schema
from .utils import (
    bucket_tenor,
    coalesce_columns,
    ensure_date_utc,
    ensure_utc,
    make_entity_id,
    normalize_ccy,
    to_float,
)


class DTCCPPDSource(PublicWebSourceBase):
    name: str = "dtcc_ppd"

    EVENTS_TABLE = "dtcc.ppd.events"
    DAILY_TABLE = "dtcc.ppd.daily"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        api_base_url: str = "https://pddata.dtcc.com/ppd/api",
        jurisdiction: str = "CFTC",
        source_mode: str = "auto",
        asset_codes: tuple[str, ...] = ("CO", "CR", "EQ", "FX", "IR"),
        list_provider: Callable[[str, str], list[dict]] | None = None,
        artifact_provider: Callable[[str], bytes] | None = None,
        now_fn: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        super().__init__(
            http_client=http_client,
            cache_dir=cache_dir,
            now_fn=now_fn,
        )
        self._api_base_url = api_base_url.rstrip("/")
        self._jurisdiction = jurisdiction.upper()
        self._jurisdiction_lower = jurisdiction.lower()
        self._source_mode = source_mode
        self._asset_codes = asset_codes
        self._list_provider = list_provider
        self._artifact_provider = artifact_provider

    def schemas(self):
        return {
            self.EVENTS_TABLE: event_table_schema(
                self.EVENTS_TABLE,
                required_columns=[
                    "asset_class",
                    "product",
                    "currency",
                    "tenor",
                    "price",
                    "notional",
                    "action",
                ],
                canonical_columns=[
                    "asset_class",
                    "product",
                    "currency",
                    "tenor",
                    "price",
                    "notional",
                    "action",
                    "trade_id",
                    "cleared",
                    "venue",
                    "effective_date",
                    "maturity_date",
                    "reported_at_utc",
                ],
                native_freq="D",
                time_semantics="point",
                release_time_column="asof_utc",
            ),
            self.DAILY_TABLE: table_schema(
                self.DAILY_TABLE,
                required_columns=[
                    "trade_count",
                    "notional_sum",
                    "price_mean",
                    "price_std",
                    "notional_median",
                    "trade_count_large",
                    "dv01_proxy_sum",
                ],
                canonical_columns=[
                    "trade_count",
                    "notional_sum",
                    "price_mean",
                    "price_std",
                    "notional_median",
                    "trade_count_large",
                    "dv01_proxy_sum",
                    "price_p10",
                    "price_p90",
                    "notional_p90",
                ],
                native_freq="D",
                time_semantics="interval_end",
            ),
        }

    def _download_file(self, file_name: str, report_type: str) -> bytes:
        if self._artifact_provider is not None:
            return self._artifact_provider(file_name)

        path = "report/intraday" if report_type == "slice" else "report/cumulative"
        url = f"{self._api_base_url}/{path}/{self._jurisdiction_lower}/{file_name}"
        return self._http.get_bytes(
            url=url,
            source="dtcc_ppd",
            artifact_name=file_name,
        )

    def _list_reports(self, report_type: str, asset_code: str) -> list[dict]:
        if self._list_provider is not None:
            return self._list_provider(report_type, asset_code)

        url = f"{self._api_base_url}/{report_type}/{self._jurisdiction}/{asset_code}"
        payload = self._http.get_bytes(
            url=url,
            source="dtcc_ppd",
            artifact_name=f"{report_type}_{self._jurisdiction}_{asset_code}.json",
        )
        parsed = json.loads(payload.decode())
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _extract_file_date(file_name: str) -> pd.Timestamp | None:
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})(?:_|\.zip$)", file_name)
        if not match:
            return None
        y, m, d = match.groups()
        return pd.Timestamp(f"{y}-{m}-{d}", tz="UTC")

    def _select_report_types(self, table: str) -> tuple[str, ...]:
        if self._source_mode == "slice":
            return ("slice",)
        if self._source_mode == "cumulative":
            return ("cumulative",)
        if table == self.DAILY_TABLE:
            return ("cumulative",)
        return ("slice", "cumulative")

    def _collect_report_files(
        self, *, start: pd.Timestamp, end: pd.Timestamp, table: str
    ) -> list[dict]:
        selected_types = self._select_report_types(table)
        records: list[dict] = []

        for report_type in selected_types:
            for asset_code in self._asset_codes:
                rows = self._list_reports(report_type, asset_code)
                for row in rows:
                    file_name = str(row.get("fileName") or "")
                    if not file_name.lower().endswith(".zip"):
                        continue

                    raw_dissem = row.get("dissemDTM")
                    if isinstance(raw_dissem, str):
                        ts = pd.Timestamp(raw_dissem)
                        dissem = (
                            ts.tz_localize("UTC")
                            if ts.tzinfo is None
                            else ts.tz_convert("UTC")
                        )
                    else:
                        dissem = pd.NaT
                    file_date = self._extract_file_date(file_name)

                    in_window = False
                    if pd.notna(dissem) and start <= dissem <= end + pd.Timedelta(
                        days=1
                    ):
                        in_window = True
                    if (
                        file_date is not None
                        and start.normalize() <= file_date <= end.normalize()
                    ):
                        in_window = True

                    if not in_window:
                        continue

                    records.append(
                        {
                            "report_type": report_type,
                            "asset_code": asset_code,
                            "file_name": file_name,
                            "dissem_dtm": dissem,
                        }
                    )

        unique: dict[tuple[str, str], dict] = {}
        for rec in records:
            key = (rec["report_type"], rec["file_name"])
            unique[key] = rec
        return list(unique.values())

    def _load_events_for_range(
        self, start: pd.Timestamp, end: pd.Timestamp, table: str
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []

        files = self._collect_report_files(start=start, end=end, table=table)
        for file_ref in files:
            payload = self._download_file(
                file_ref["file_name"],
                report_type=file_ref["report_type"],
            )
            parsed = parse_zip_csv_bytes(payload)
            if not parsed.empty:
                parsed["_dtcc_file_name"] = file_ref["file_name"]
                parsed["_dtcc_report_type"] = file_ref["report_type"]
                parts.append(parsed)
        if not parts:
            return pd.DataFrame()
        raw = pd.concat(parts, ignore_index=True)
        return self._normalize_events(raw)

    def _normalize_events(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)

        reported_raw = coalesce_columns(
            raw,
            [
                "reported_at_utc",
                "reported_at",
                "dissemination_timestamp",
                "report_timestamp",
                "report_time",
            ],
        )
        ts_raw = coalesce_columns(
            raw,
            [
                "execution_timestamp",
                "execution_time",
                "event_timestamp",
                "timestamp",
                "trade_timestamp",
            ],
        )

        out["reported_at_utc"] = ensure_utc(reported_raw)
        out["ts_utc"] = ensure_utc(ts_raw).fillna(out["reported_at_utc"])

        out["asset_class"] = coalesce_columns(
            raw, ["asset_class", "assetclass"]
        ).fillna("unknown")
        out["product"] = coalesce_columns(raw, ["product", "product_type"]).fillna(
            "unknown"
        )

        out["currency"] = (
            coalesce_columns(raw, ["currency", "notional_currency", "ccy"])
            .map(normalize_ccy)
            .fillna("unk")
        )

        effective = ensure_date_utc(
            coalesce_columns(raw, ["effective_date", "start_date"])
        )
        maturity = ensure_date_utc(
            coalesce_columns(raw, ["maturity_date", "end_date", "termination_date"])
        )

        tenor_raw = coalesce_columns(raw, ["tenor", "maturity_tenor", "term"]).astype(
            str
        )
        maturity_years = (maturity - effective).map(
            lambda td: td / pd.Timedelta(days=365.25) if pd.notna(td) else float("nan")
        )
        tenor_bucket = tenor_raw.map(bucket_tenor)
        inferred = maturity_years.map(
            lambda x: bucket_tenor(float(x)) if pd.notna(x) else "unk"
        )
        out["tenor"] = tenor_bucket.where(tenor_bucket != "unk", inferred)

        out["price"] = to_float(
            coalesce_columns(raw, ["price", "fixed_rate", "spread", "premium"])
        )
        out["notional"] = to_float(
            coalesce_columns(raw, ["notional", "notional_amount", "amount"])
        )
        out["action"] = (
            coalesce_columns(raw, ["action", "event_type"])
            .fillna("unknown")
            .astype(str)
        )

        out["trade_id"] = coalesce_columns(
            raw, ["trade_id", "dissemination_id"]
        ).astype(str)
        out["venue"] = coalesce_columns(raw, ["venue", "sef"]).fillna("").astype(str)
        out["effective_date"] = effective
        out["maturity_date"] = maturity

        cleared = (
            coalesce_columns(raw, ["cleared", "is_cleared"]).astype(str).str.lower()
        )
        out["cleared"] = cleared.isin({"1", "true", "yes", "y"})

        out["asof_utc"] = out["reported_at_utc"].fillna(self._now_utc())

        asset_norm = (
            out["asset_class"]
            .astype(str)
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        product_norm = (
            out["product"].astype(str).str.lower().str.replace(" ", "_", regex=False)
        )
        out["entity_id"] = [
            make_entity_id(
                "dtccppd",
                asset if asset and asset != "unknown" else "unk",
                product if product and product != "unknown" else "unk",
                ccy if ccy else "unk",
                tenor if tenor else "unk",
            )
            for asset, product, ccy, tenor in zip(
                asset_norm, product_norm, out["currency"], out["tenor"]
            )
        ]

        return out

    def _aggregate_daily(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(columns=["date", "entity_id", "asof_utc"])

        work = events.copy()
        work["date"] = ensure_date_utc(work["ts_utc"])

        duration_map = {
            "2y": 1.9,
            "5y": 4.5,
            "10y": 8.5,
            "30y": 20.0,
        }
        work["duration_scalar"] = work["tenor"].map(duration_map).fillna(1.0)
        work["dv01_proxy"] = (
            work["notional"].fillna(0.0) * work["duration_scalar"] * 1e-4
        )

        grouped = work.groupby(["date", "entity_id"], as_index=False)
        daily = grouped.agg(
            trade_count=("entity_id", "size"),
            notional_sum=("notional", "sum"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            notional_median=("notional", "median"),
            trade_count_large=(
                "notional",
                lambda x: int((x.fillna(0) > 1_000_000).sum()),
            ),
            dv01_proxy_sum=("dv01_proxy", "sum"),
            price_p10=("price", lambda x: x.quantile(0.1) if x.notna().any() else None),
            price_p90=("price", lambda x: x.quantile(0.9) if x.notna().any() else None),
            notional_p90=(
                "notional",
                lambda x: x.quantile(0.9) if x.notna().any() else None,
            ),
            asof_utc=("asof_utc", "max"),
        )
        return daily

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table not in {self.EVENTS_TABLE, self.DAILY_TABLE}:
            raise ValueError(f"Unknown table: {q.table}")

        now_utc = self._now_utc()
        start = q.start or (now_utc.normalize() - pd.Timedelta(days=7))
        end = q.end or now_utc

        events = self._load_events_for_range(start=start, end=end, table=q.table)
        schema = self._schema(q.table)
        if events.empty:
            return self._empty_frame(
                schema,
                time_col=schema.time_column,
                entity_col=schema.entity_column,
            )

        if q.table == self.EVENTS_TABLE:
            return self._finalize(
                events,
                q=q,
                schema=schema,
                time_col="ts_utc",
                entity_col="entity_id",
                sort_by=[],
            )

        daily = self._aggregate_daily(events)
        return self._finalize(
            daily,
            q=q,
            schema=schema,
            sort_by=[],
        )
