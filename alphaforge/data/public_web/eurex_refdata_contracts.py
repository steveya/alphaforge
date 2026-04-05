from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .schema_helpers import table_schema
from .tabular import resolved_text_series
from .utils import ensure_date_utc, first_existing, make_entity_id


class EurexRefdataContractsSource(PublicWebSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
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
                native_freq="D",
                time_semantics="point",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)

        payload = self._http.get_bytes(
            url=self._api_url,
            source="eurex_refdata_contracts",
            artifact_name=Path(self._api_url.split("?")[0]).name or "refdata.json",
        )

        data = json.loads(payload.decode())
        records = data if isinstance(data, list) else data.get("data", [])
        frame = pd.DataFrame(records)

        schema = self._schema()
        if frame.empty:
            return self._empty_frame(schema, entity_col="entity_id")

        out = pd.DataFrame(index=frame.index)
        out["symbol"] = resolved_text_series(
            frame,
            ["symbol", "contract_symbol"],
            default="",
        )
        out["product_name"] = resolved_text_series(
            frame,
            ["product_name", "product"],
            default="",
        )
        out["product_group"] = resolved_text_series(
            frame,
            ["product_group", "group"],
            default="",
        )
        out["currency"] = (
            frame[first_existing(frame, "currency", "ccy")].astype(str).str.lower()
        )
        out["expiry_date"] = ensure_date_utc(
            frame[first_existing(frame, "expiry_date", "expiry", "maturity_date")]
        )

        for col in ["underlying", "multiplier", "tick_size", "isin"]:
            src = first_existing(frame, col)
            out[col] = frame[src] if src else pd.NA

        snapshot = self._asof_utc(q)
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

        return self._finalize(out, q=q, schema=schema, sort_by=[])
