"""CFTC Commitments of Traders public-web sources.

Includes:

- ``CFTCCoTSource`` for Traders in Financial Futures (TFF; futures only)
- ``CFTCDisaggregatedCoTSource`` for disaggregated commodity futures

Canonical entity-id pattern::

    futures.{contract_code}.{trader_category}.cftc
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .archive import (
    iter_yearly_archive_fetches,
    iter_yearly_archive_urls,
    read_zip_members,
)
from .http import CachedHttpClient
from .parsing import normalize_headers
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    make_entity_id,
    project_columns,
    to_float,
)

_TraderCategorySpec = tuple[str, str, str | None, str | None, str | None, str]

# Common column names after header normalisation
_COL_REPORT_DATE = "report_date_as_yyyy_mm_dd"
_COL_REPORT_DATE_LEGACY = "report_date_as_mm_dd_yyyy"
_COL_MARKET = "market_and_exchange_names"
_COL_CFTC_CODE = "cftc_contract_market_code"
_COL_OI = "open_interest_all"

# TFF columns
_COL_LEV_LONG = "lev_money_positions_long_all"
_COL_LEV_SHORT = "lev_money_positions_short_all"
_COL_LEV_SPREAD = "lev_money_positions_spread_all"
_COL_CHG_LEV_LONG = "change_in_lev_money_long_all"
_COL_CHG_LEV_SHORT = "change_in_lev_money_short_all"

_COL_DEALER_LONG = "dealer_positions_long_all"
_COL_DEALER_SHORT = "dealer_positions_short_all"
_COL_DEALER_SPREAD = "dealer_positions_spread_all"
_COL_CHG_DEALER_LONG = "change_in_dealer_long_all"
_COL_CHG_DEALER_SHORT = "change_in_dealer_short_all"

_COL_AM_LONG = "asset_mgr_positions_long_all"
_COL_AM_SHORT = "asset_mgr_positions_short_all"
_COL_AM_SPREAD = "asset_mgr_positions_spread_all"
_COL_CHG_AM_LONG = "change_in_asset_mgr_long_all"
_COL_CHG_AM_SHORT = "change_in_asset_mgr_short_all"

_COL_OTHER_LONG = "other_rept_positions_long_all"
_COL_OTHER_SHORT = "other_rept_positions_short_all"
_COL_OTHER_SPREAD = "other_rept_positions_spread_all"
_COL_CHG_OTHER_LONG = "change_in_other_rept_long_all"
_COL_CHG_OTHER_SHORT = "change_in_other_rept_short_all"

# Disaggregated columns
_COL_PROD_MERC_LONG = "prod_merc_positions_long_all"
_COL_PROD_MERC_SHORT = "prod_merc_positions_short_all"
_COL_CHG_PROD_MERC_LONG = "change_in_prod_merc_long_all"
_COL_CHG_PROD_MERC_SHORT = "change_in_prod_merc_short_all"

_COL_SWAP_LONG = "swap_positions_long_all"
_COL_SWAP_SHORT = "swap_positions_short_all"
_COL_SWAP_SPREAD = "swap_positions_spread_all"
_COL_CHG_SWAP_LONG = "change_in_swap_long_all"
_COL_CHG_SWAP_SHORT = "change_in_swap_short_all"

_COL_M_MONEY_LONG = "m_money_positions_long_all"
_COL_M_MONEY_SHORT = "m_money_positions_short_all"
_COL_M_MONEY_SPREAD = "m_money_positions_spread_all"
_COL_CHG_M_MONEY_LONG = "change_in_m_money_long_all"
_COL_CHG_M_MONEY_SHORT = "change_in_m_money_short_all"

_TFF_TRADER_CATEGORIES: dict[str, _TraderCategorySpec] = {
    "lev_money": (
        _COL_LEV_LONG,
        _COL_LEV_SHORT,
        _COL_LEV_SPREAD,
        _COL_CHG_LEV_LONG,
        _COL_CHG_LEV_SHORT,
        "lev_money",
    ),
    "dealer": (
        _COL_DEALER_LONG,
        _COL_DEALER_SHORT,
        _COL_DEALER_SPREAD,
        _COL_CHG_DEALER_LONG,
        _COL_CHG_DEALER_SHORT,
        "dealer",
    ),
    "asset_mgr": (
        _COL_AM_LONG,
        _COL_AM_SHORT,
        _COL_AM_SPREAD,
        _COL_CHG_AM_LONG,
        _COL_CHG_AM_SHORT,
        "asset_mgr",
    ),
    "other_rept": (
        _COL_OTHER_LONG,
        _COL_OTHER_SHORT,
        _COL_OTHER_SPREAD,
        _COL_CHG_OTHER_LONG,
        _COL_CHG_OTHER_SHORT,
        "other_rept",
    ),
}

_DISAGG_TRADER_CATEGORIES: dict[str, _TraderCategorySpec] = {
    "prod_merc": (
        _COL_PROD_MERC_LONG,
        _COL_PROD_MERC_SHORT,
        None,
        _COL_CHG_PROD_MERC_LONG,
        _COL_CHG_PROD_MERC_SHORT,
        "prod_merc",
    ),
    "swap": (
        _COL_SWAP_LONG,
        _COL_SWAP_SHORT,
        _COL_SWAP_SPREAD,
        _COL_CHG_SWAP_LONG,
        _COL_CHG_SWAP_SHORT,
        "swap",
    ),
    "m_money": (
        _COL_M_MONEY_LONG,
        _COL_M_MONEY_SHORT,
        _COL_M_MONEY_SPREAD,
        _COL_CHG_M_MONEY_LONG,
        _COL_CHG_M_MONEY_SHORT,
        "m_money",
    ),
    "other_rept": (
        _COL_OTHER_LONG,
        _COL_OTHER_SHORT,
        _COL_OTHER_SPREAD,
        _COL_CHG_OTHER_LONG,
        _COL_CHG_OTHER_SHORT,
        "other_rept",
    ),
}

_TFF_CONTRACT_CODES: dict[str, str] = {
    "1170E1": "vix",
    "13874+": "sp500",
    "13874A": "sp500_e_mini",
    "33874E": "sp500_micro",
    "209742": "vix",
    "099741": "eur",
    "096742": "gbp",
    "097741": "jpy",
    "092741": "chf",
    "090741": "cad",
    "232741": "aud",
    "112741": "nzd",
    "095741": "mxn",
    "089741": "sek",
    "088741": "nok",
    "13874P": "sofr_3m",
    "134741": "ust_10y",
    "020601": "ust_30y",
    "044601": "ust_5y",
    "042601": "ust_2y",
    "043602": "fed_funds",
}

_TFF_MARKET_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bVIX\b", re.IGNORECASE), "vix"),
    (re.compile(r"\bCBOE VOLATILITY INDEX\b", re.IGNORECASE), "vix"),
    (re.compile(r"\bS&P 500\b", re.IGNORECASE), "sp500"),
    (re.compile(r"\bEURO FX\b", re.IGNORECASE), "eur"),
    (re.compile(r"\bBRITISH POUND\b", re.IGNORECASE), "gbp"),
    (re.compile(r"\bJAPANESE YEN\b", re.IGNORECASE), "jpy"),
    (re.compile(r"\bSWISS FRANC\b", re.IGNORECASE), "chf"),
    (re.compile(r"\bCANADIAN DOLLAR\b", re.IGNORECASE), "cad"),
    (re.compile(r"\bAUSTRALIAN DOLLAR\b", re.IGNORECASE), "aud"),
    (re.compile(r"\bNEW ZEALAND DOLLAR\b|\bNZ DOLLAR\b", re.IGNORECASE), "nzd"),
    (re.compile(r"\bSOFR\b", re.IGNORECASE), "sofr_3m"),
    (re.compile(r"\b10.YEAR\b.*\bT.NOTE\b", re.IGNORECASE), "ust_10y"),
]

_DISAGG_MARKET_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bWHEAT[- ]SRW\b", re.IGNORECASE), "wheat_srw"),
    (re.compile(r"\bWHEAT[- ]HRW\b", re.IGNORECASE), "wheat_hrw"),
    (re.compile(r"\bCORN\b", re.IGNORECASE), "corn"),
    (re.compile(r"\bSOYBEAN(S)?\b", re.IGNORECASE), "soybeans"),
    (re.compile(r"\bSOYBEAN OIL\b", re.IGNORECASE), "soybean_oil"),
    (re.compile(r"\bSOYBEAN MEAL\b", re.IGNORECASE), "soybean_meal"),
    (re.compile(r"\bLIGHT SWEET CRUDE OIL\b|\bWTI\b", re.IGNORECASE), "wti"),
    (re.compile(r"\bBRENT CRUDE OIL\b", re.IGNORECASE), "brent"),
    (re.compile(r"\bNATURAL GAS\b", re.IGNORECASE), "natgas"),
    (re.compile(r"\bGOLD\b", re.IGNORECASE), "gold"),
    (re.compile(r"\bSILVER\b", re.IGNORECASE), "silver"),
    (re.compile(r"\bCOPPER\b", re.IGNORECASE), "copper"),
    (re.compile(r"\bCOFFEE\b", re.IGNORECASE), "coffee"),
    (re.compile(r"\bSUGAR\b", re.IGNORECASE), "sugar"),
    (re.compile(r"\bCOTTON\b", re.IGNORECASE), "cotton"),
]


def _slugify_contract_name(market_name: str) -> str:
    """Turn the human-readable market name into a stable identifier."""
    primary = (market_name or "").split(" - ", 1)[0].strip()
    slug = re.sub(r"[^a-z0-9]+", "_", primary.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "unknown"


def _infer_contract_code(
    cftc_code: str,
    market_name: str,
    *,
    contract_codes: dict[str, str],
    market_name_patterns: Sequence[tuple[re.Pattern[str], str]],
    prefer_market_slug: bool = False,
) -> str:
    """Map a CFTC contract code or market name to a short identifier."""
    code = str(cftc_code).strip()
    if code in contract_codes:
        return contract_codes[code]
    for pat, name in market_name_patterns:
        if pat.search(market_name or ""):
            return name
    if prefer_market_slug:
        market_slug = _slugify_contract_name(market_name)
        if market_slug != "unknown":
            return market_slug
    return re.sub(r"[^a-z0-9]", "", code.lower()) or "unknown"


def _publication_date(report_date: pd.Series) -> pd.Series:
    """Compute publication date (Friday) from report date (Tuesday)."""
    offset = pd.tseries.offsets.BDay(3)
    return report_date.map(lambda d: d + offset if pd.notna(d) else pd.NaT)


def _parse_yymmdd_dates(series: pd.Series) -> pd.Series:
    """Parse CFTC YYMMDD date fields reliably, preserving leading zeros."""
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )
    parsed = pd.to_datetime(text, format="%y%m%d", errors="coerce", utc=True)
    normalized = parsed.map(lambda ts: ts.normalize() if pd.notna(ts) else pd.NaT)
    return pd.to_datetime(normalized, utc=True)


def _parse_report_dates(frame: pd.DataFrame) -> pd.Series:
    """Parse the report-date column across CFTC's header variants."""
    for candidate in (_COL_REPORT_DATE, _COL_REPORT_DATE_LEGACY, "report_date"):
        if candidate in frame.columns:
            return ensure_date_utc(frame[candidate])
    if "as_of_date_in_form_yymmdd" in frame.columns:
        return _parse_yymmdd_dates(frame["as_of_date_in_form_yymmdd"])
    return pd.to_datetime(pd.Series([pd.NaT] * len(frame), index=frame.index), utc=True)


class _BaseCFTCCoTSource(DataSource):
    """Shared implementation for CFTC CoT ZIP archives."""

    name: str = "cftc_cot"
    TABLE = "cftc.cot"
    URL_TEMPLATE = ""
    FIRST_YEAR = 2006
    YEARLY_FIRST_YEAR = 2006
    HISTORICAL_URL: str | None = None
    HISTORICAL_LAST_YEAR: int | None = None
    TRADER_CATEGORIES: dict[str, _TraderCategorySpec] = {}
    CONTRACT_CODES: dict[str, str] = {}
    MARKET_NAME_PATTERNS: Sequence[tuple[re.Pattern[str], str]] = ()
    PREFER_MARKET_SLUG_FALLBACK = False

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        url_template: str | None = None,
        file_urls: list[str] | None = None,
        trader_categories: Sequence[str] | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._url_template = url_template or self.URL_TEMPLATE
        self._file_urls = file_urls
        self._trader_categories = (
            list(trader_categories) if trader_categories else list(self.TRADER_CATEGORIES)
        )

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=[
                    "long_positions",
                    "short_positions",
                ],
                canonical_columns=[
                    "long_positions",
                    "short_positions",
                    "spread_positions",
                    "open_interest",
                    "change_long",
                    "change_short",
                    "trader_category",
                    "contract_code",
                    "publication_date",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq="W",
                time_semantics="interval_end",
            )
        }

    def _year_urls(self, start_year: int, end_year: int) -> list[str]:
        return iter_yearly_archive_urls(
            start_year=start_year,
            end_year=end_year,
            url_template=self._url_template,
            first_year=self.FIRST_YEAR,
            yearly_first_year=self.YEARLY_FIRST_YEAR,
            historical_url=self.HISTORICAL_URL,
            historical_last_year=self.HISTORICAL_LAST_YEAR,
            file_urls=self._file_urls,
        )

    def _year_fetches(self, start_year: int, end_year: int):
        return iter_yearly_archive_fetches(
            start_year=start_year,
            end_year=end_year,
            url_template=self._url_template,
            first_year=self.FIRST_YEAR,
            yearly_first_year=self.YEARLY_FIRST_YEAR,
            historical_url=self.HISTORICAL_URL,
            historical_last_year=self.HISTORICAL_LAST_YEAR,
            file_urls=self._file_urls,
            fallback_artifact_prefix=self.name,
        )

    def _read_zip(self, planned) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=planned.url,
            source=self.name,
            artifact_name=planned.artifact_name,
        )

        frames: list[pd.DataFrame] = []
        for _name, member_payload in read_zip_members(payload, suffixes=(".csv", ".txt")):
            frames.append(normalize_headers(pd.read_csv(io.BytesIO(member_payload))))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _to_long(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()

        report_dates = _parse_report_dates(frame)
        if report_dates.isna().all():
            return pd.DataFrame()
        cftc_codes = (
            frame[_COL_CFTC_CODE].astype(str)
            if _COL_CFTC_CODE in frame.columns
            else pd.Series("", index=frame.index)
        )
        market_names = (
            frame[_COL_MARKET].astype(str)
            if _COL_MARKET in frame.columns
            else pd.Series("", index=frame.index)
        )
        contract_codes = [
            _infer_contract_code(
                code,
                name,
                contract_codes=self.CONTRACT_CODES,
                market_name_patterns=self.MARKET_NAME_PATTERNS,
                prefer_market_slug=self.PREFER_MARKET_SLUG_FALLBACK,
            )
            for code, name in zip(cftc_codes, market_names)
        ]

        oi = (
            to_float(frame[_COL_OI])
            if _COL_OI in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )

        rows: list[pd.DataFrame] = []
        for cat_key in self._trader_categories:
            if cat_key not in self.TRADER_CATEGORIES:
                continue

            col_long, col_short, col_spread, col_chg_long, col_chg_short, label = (
                self.TRADER_CATEGORIES[cat_key]
            )
            if col_long not in frame.columns or col_short not in frame.columns:
                continue

            chunk = pd.DataFrame(
                {
                    "report_date": report_dates,
                    "contract_code": contract_codes,
                    "trader_category": label,
                    "long_positions": to_float(frame[col_long]),
                    "short_positions": to_float(frame[col_short]),
                    "spread_positions": (
                        to_float(frame[col_spread])
                        if col_spread is not None and col_spread in frame.columns
                        else float("nan")
                    ),
                    "open_interest": oi,
                    "change_long": (
                        to_float(frame[col_chg_long])
                        if col_chg_long is not None and col_chg_long in frame.columns
                        else float("nan")
                    ),
                    "change_short": (
                        to_float(frame[col_chg_short])
                        if col_chg_short is not None and col_chg_short in frame.columns
                        else float("nan")
                    ),
                }
            )
            chunk["entity_id"] = [
                make_entity_id("futures", contract_code, label, "cftc")
                for contract_code in chunk["contract_code"]
            ]
            rows.append(chunk)

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)
        out["publication_date"] = _publication_date(out["report_date"])
        out["date"] = out["publication_date"]
        out["asof_utc"] = out["publication_date"]
        return out

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        schema = self.schemas()[self.TABLE]
        start_year = q.start.year if q.start is not None else self.FIRST_YEAR
        end_year = q.end.year if q.end is not None else pd.Timestamp.now(tz="UTC").year

        frames: list[pd.DataFrame] = []
        for planned in self._year_fetches(start_year, end_year):
            try:
                raw = self._read_zip(planned)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load CFTC archive {planned.url} "
                    f"for source '{self.name}'."
                ) from exc
            long = self._to_long(raw)
            if not long.empty:
                frames.append(long)

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
        out = out.drop_duplicates(ignore_index=True)
        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)


class CFTCCoTSource(_BaseCFTCCoTSource):
    """CFTC Commitments of Traders: Traders in Financial Futures (futures only)."""

    name = "cftc_cot"
    TABLE = "cftc.cot.tff"
    URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    FIRST_YEAR = 2006
    YEARLY_FIRST_YEAR = 2010
    HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip"
    HISTORICAL_LAST_YEAR = 2016
    TRADER_CATEGORIES = _TFF_TRADER_CATEGORIES
    CONTRACT_CODES = _TFF_CONTRACT_CODES
    MARKET_NAME_PATTERNS = _TFF_MARKET_NAME_PATTERNS


class CFTCDisaggregatedCoTSource(_BaseCFTCCoTSource):
    """CFTC Commitments of Traders: disaggregated commodity futures."""

    name = "cftc_cot_disagg"
    TABLE = "cftc.cot.disagg"
    URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    FIRST_YEAR = 2006
    YEARLY_FIRST_YEAR = 2010
    HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip"
    HISTORICAL_LAST_YEAR = 2016
    TRADER_CATEGORIES = _DISAGG_TRADER_CATEGORIES
    MARKET_NAME_PATTERNS = _DISAGG_MARKET_NAME_PATTERNS
    PREFER_MARKET_SLUG_FALLBACK = True
