"""Federal Reserve Board three-factor term-structure benchmark loader."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import (
    apply_query_filters,
    bucket_tenor,
    ensure_date_utc,
    make_entity_id,
    project_columns,
)


class FRBTermStructureBenchmarkSource(DataSource):
    """Federal Reserve Board Kim-Wright three-factor benchmark series."""

    name: str = "frb_term_structure"
    TABLE = "frb.term_structure"
    CSV_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200533.csv"

    _MNEMONIC_PATTERN = re.compile(
        r"^THREE(?P<family>FF|FY)(?P<term_premium>TP)?(?P<maturity>\d{2})00\.B$"
    )

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        csv_url: str | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._csv_url = csv_url or self.CSV_URL

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=["value", "mnemonic", "category", "maturity_years"],
                entity_column="entity_id",
                time_column="date",
                native_freq="B",
                time_semantics="point",
            )
        }

    @classmethod
    def _parse_mnemonic(cls, mnemonic: str) -> tuple[str, float] | None:
        match = cls._MNEMONIC_PATTERN.match(str(mnemonic).strip().upper())
        if match is None:
            return None

        family = match.group("family")
        is_term_premium = match.group("term_premium") is not None
        maturity_years = float(match.group("maturity"))

        if family == "FY":
            category = "yield_term_premium" if is_term_premium else "yield"
        else:
            category = "forward_term_premium" if is_term_premium else "forward_rate"
        return category, maturity_years

    def _load_raw(self) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=self._csv_url,
            source="frb_term_structure",
            artifact_name="feds200533.csv",
        )
        return pd.read_csv(io.BytesIO(payload))

    def _to_long(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "mnemonic",
                    "category",
                    "maturity_years",
                    "value",
                    "entity_id",
                    "asof_utc",
                ]
            )

        date_column = raw.columns[0]
        long = raw.rename(columns={date_column: "date"}).melt(
            id_vars=["date"],
            var_name="mnemonic",
            value_name="value",
        )
        long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=True)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["date", "value"]).copy()

        parsed = long["mnemonic"].map(self._parse_mnemonic)
        long = long[parsed.notna()].copy()
        long["category"] = parsed.map(lambda item: item[0])
        long["maturity_years"] = parsed.map(lambda item: item[1])
        long["date"] = ensure_date_utc(long["date"])
        long["entity_id"] = [
            make_entity_id(
                "rates",
                "us",
                "frb",
                category,
                bucket_tenor(maturity),
            )
            for category, maturity in zip(long["category"], long["maturity_years"])
        ]
        long["asof_utc"] = long["date"]
        return long.sort_values(["date", "mnemonic"]).reset_index(drop=True)

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        long = self._to_long(self._load_raw())
        long = apply_query_filters(long, q=q, time_col="date", entity_col="entity_id")
        return project_columns(
            long,
            required_columns=["value"],
            requested_columns=q.columns,
            time_col="date",
            entity_col="entity_id",
        )

    def fetch_wide(self, q: Query, *, category: str | None = None) -> pd.DataFrame:
        df = self.fetch(
            Query(
                table=q.table,
                columns=["value", "category", "maturity_years"],
                start=q.start,
                end=q.end,
                entities=q.entities,
                asof=q.asof,
            )
        )
        if category is not None:
            df = df[df["category"] == category]
        wide = df.pivot_table(
            index="date",
            columns="maturity_years",
            values="value",
            aggfunc="last",
        )
        return wide.sort_index()
