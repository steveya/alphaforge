from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query

from .http import CachedHttpClient
from .schema_helpers import table_schema
from .tabular import (
    TabularDocumentSourceBase,
    resolved_text_series,
)
from .utils import first_existing, make_entity_id


class CMEProductSlateSource(TabularDocumentSourceBase):
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
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._csv_url = csv_url or self.URL

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
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
                native_freq="D",
                time_semantics="point",
            )
        }

    def _download(self) -> pd.DataFrame:
        return self._read_csv_frame(
            url=self._csv_url,
            source="cme_productslate",
            artifact_name="productslate.csv",
        )

    def _build_entity_id(self, row: pd.Series) -> str:
        product_code = str(row.get("product_code") or "unk").lower()
        asset_class = str(row.get("asset_class") or "").lower()
        domain = (
            "rates" if "rate" in asset_class or "interest" in asset_class else "other"
        )
        instrument = "opt" if "option" in asset_class else "fut"
        return make_entity_id(domain, instrument, product_code, "cme")

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)

        schema = self._schema()
        raw = self._download()

        out = pd.DataFrame()
        out["exchange"] = resolved_text_series(
            raw,
            ["exchange"],
            default="",
        )
        out["product_code"] = resolved_text_series(
            raw,
            ["product_code", "product", "code"],
            default="",
        )
        out["product_name"] = resolved_text_series(
            raw,
            ["product_name", "name", "description"],
            default="",
        )

        asset_col = first_existing(raw, "asset_class", "assetclass", "asset")
        sub_asset_col = first_existing(
            raw, "sub_asset_class", "subassetclass", "sub_asset"
        )
        out["asset_class"] = raw[asset_col].astype(str) if asset_col else ""
        out["sub_asset_class"] = raw[sub_asset_col].astype(str) if sub_asset_col else ""

        for optional_col in ["globex_symbol", "clearing_code", "mic"]:
            src_col = first_existing(raw, optional_col)
            out[optional_col] = raw[src_col].astype(str) if src_col else ""

        now_utc = self._asof_utc(q)
        out["date"] = now_utc.normalize()
        out["asof_utc"] = now_utc
        out["entity_id"] = out.apply(self._build_entity_id, axis=1)

        return self._finalize(out, q=q, schema=schema, sort_by=[])
