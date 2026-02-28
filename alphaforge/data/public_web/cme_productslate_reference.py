from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .parsing import parse_csv_bytes
from .utils import (
    apply_query_filters,
    first_existing,
    make_entity_id,
    project_columns,
)


class CMEProductSlateSource(DataSource):
    name: str = "cme_productslate"

    TABLE = "cme.productslate.reference"
    URL = "https://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.csv"

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        csv_url: str | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._csv_url = csv_url or self.URL

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=[
                    "exchange",
                    "product_code",
                    "product_name",
                    "asset_class",
                    "sub_asset_class",
                ],
                canonical_columns=[
                    "exchange",
                    "product_code",
                    "product_name",
                    "asset_class",
                    "sub_asset_class",
                    "globex_symbol",
                    "clearing_code",
                    "mic",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq="D",
                time_semantics="point",
            )
        }

    def _download(self) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=self._csv_url,
            source="cme_productslate",
            artifact_name="productslate.csv",
        )
        return parse_csv_bytes(payload)

    def _build_entity_id(self, row: pd.Series) -> str:
        product_code = str(row.get("product_code") or "unk").lower()
        asset_class = str(row.get("asset_class") or "").lower()
        domain = (
            "rates" if "rate" in asset_class or "interest" in asset_class else "other"
        )
        instrument = "opt" if "option" in asset_class else "fut"
        return make_entity_id(domain, instrument, product_code, "cme")

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        schema = self.schemas()[q.table]
        raw = self._download()
        raw_columns = {c.lower(): c for c in raw.columns}

        out = pd.DataFrame()
        out["exchange"] = raw[
            first_existing(raw, "exchange") or raw_columns.get("exchange", "exchange")
        ]
        out["product_code"] = raw[
            first_existing(raw, "product_code", "product", "code")
            or raw_columns.get("product_code", "product_code")
        ].astype(str)
        out["product_name"] = raw[
            first_existing(raw, "product_name", "name", "description")
            or raw_columns.get("product_name", "product_name")
        ].astype(str)

        asset_col = first_existing(raw, "asset_class", "assetclass", "asset")
        sub_asset_col = first_existing(
            raw, "sub_asset_class", "subassetclass", "sub_asset"
        )
        out["asset_class"] = raw[asset_col].astype(str) if asset_col else ""
        out["sub_asset_class"] = raw[sub_asset_col].astype(str) if sub_asset_col else ""

        for optional_col in ["globex_symbol", "clearing_code", "mic"]:
            src_col = first_existing(raw, optional_col)
            out[optional_col] = raw[src_col].astype(str) if src_col else ""

        now_utc = pd.Timestamp.now(tz="UTC")
        out["date"] = now_utc.normalize()
        out["asof_utc"] = now_utc
        out["entity_id"] = out.apply(self._build_entity_id, axis=1)

        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
