from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    first_existing,
    make_entity_id,
    project_columns,
)


class EurexRefdataContractsSource(DataSource):
    name: str = "eurex_refdata_contracts"
    TABLE = "eurex.refdata.contracts"

    def __init__(
        self,
        *,
        api_url: str,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._api_url = api_url
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=[
                    "symbol",
                    "product_name",
                    "product_group",
                    "currency",
                    "expiry_date",
                ],
                canonical_columns=[
                    "symbol",
                    "product_name",
                    "product_group",
                    "currency",
                    "expiry_date",
                    "underlying",
                    "multiplier",
                    "tick_size",
                    "isin",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq="D",
                time_semantics="point",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        payload = self._http.get_bytes(
            url=self._api_url,
            source="eurex_refdata_contracts",
            artifact_name=Path(self._api_url.split("?")[0]).name or "refdata.json",
        )

        data = json.loads(payload.decode())
        records = data if isinstance(data, list) else data.get("data", [])
        frame = pd.DataFrame(records)

        schema = self.schemas()[self.TABLE]
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    schema.time_column,
                    schema.entity_column,
                    "asof_utc",
                    *schema.required_columns,
                ]
            )

        out = pd.DataFrame(index=frame.index)
        out["symbol"] = frame[
            first_existing(frame, "symbol", "contract_symbol")
        ].astype(str)
        out["product_name"] = frame[
            first_existing(frame, "product_name", "product")
        ].astype(str)
        out["product_group"] = frame[
            first_existing(frame, "product_group", "group")
        ].astype(str)
        out["currency"] = (
            frame[first_existing(frame, "currency", "ccy")].astype(str).str.lower()
        )
        out["expiry_date"] = ensure_date_utc(
            frame[first_existing(frame, "expiry_date", "expiry", "maturity_date")]
        )

        for col in ["underlying", "multiplier", "tick_size", "isin"]:
            src = first_existing(frame, col)
            out[col] = frame[src] if src else pd.NA

        snapshot = pd.Timestamp.now(tz="UTC")
        out["date"] = snapshot.normalize()
        out["asof_utc"] = snapshot

        out["entity_id"] = [
            make_entity_id(
                "rates",
                "fut",
                symbol,
                expiry.strftime("%Y%m") if pd.notna(expiry) else "unk",
                "eurex",
            )
            for symbol, expiry in zip(out["symbol"].str.lower(), out["expiry_date"])
        ]

        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
