"""CFTC Commitments of Traders — Traders in Financial Futures (TFF).

Downloads and normalises the disaggregated Traders in Financial Futures
report from the CFTC bulk-file archive.  Each year's data comes as a ZIP
containing a single CSV with positions as of Tuesday, published the
following Friday.

Canonical entity-id pattern::

    futures.{contract_code}.{trader_category}.cftc

Example entity IDs::

    futures.vix.lev_money.cftc
    futures.vix.dealer.cftc
    futures.sp500.asset_mgr.cftc
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .parsing import normalize_headers
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    make_entity_id,
    project_columns,
    to_float,
)

# ── Column name mapping (snake_case after normalisation) ───────────────
_COL_REPORT_DATE = "report_date_as_yyyy_mm_dd"
_COL_MARKET = "market_and_exchange_names"
_COL_CFTC_CODE = "cftc_contract_market_code"

# Leveraged-money columns
_COL_LEV_LONG = "lev_money_positions_long_all"
_COL_LEV_SHORT = "lev_money_positions_short_all"
_COL_LEV_SPREAD = "lev_money_positions_spread_all"
_COL_CHG_LEV_LONG = "change_in_lev_money_long_all"
_COL_CHG_LEV_SHORT = "change_in_lev_money_short_all"

# Dealer columns
_COL_DEALER_LONG = "dealer_positions_long_all"
_COL_DEALER_SHORT = "dealer_positions_short_all"
_COL_DEALER_SPREAD = "dealer_positions_spread_all"
_COL_CHG_DEALER_LONG = "change_in_dealer_long_all"
_COL_CHG_DEALER_SHORT = "change_in_dealer_short_all"

# Asset-manager columns
_COL_AM_LONG = "asset_mgr_positions_long_all"
_COL_AM_SHORT = "asset_mgr_positions_short_all"
_COL_AM_SPREAD = "asset_mgr_positions_spread_all"
_COL_CHG_AM_LONG = "change_in_asset_mgr_long_all"
_COL_CHG_AM_SHORT = "change_in_asset_mgr_short_all"

# Other-reportable columns
_COL_OTHER_LONG = "other_rept_positions_long_all"
_COL_OTHER_SHORT = "other_rept_positions_short_all"
_COL_OTHER_SPREAD = "other_rept_positions_spread_all"
_COL_CHG_OTHER_LONG = "change_in_other_rept_long_all"
_COL_CHG_OTHER_SHORT = "change_in_other_rept_short_all"

_COL_OI = "open_interest_all"

# Each trader category → (long, short, spread, chg_long, chg_short, label)
_TRADER_CATEGORIES: dict[str, tuple[str, str, str, str, str, str]] = {
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

# Well-known CFTC contract codes → short name.
_CONTRACT_CODES: dict[str, str] = {
    "1170E1": "vix",
    "13874+": "sp500",
    "13874A": "sp500_e_mini",
    "33874E": "sp500_micro",
    "209742": "vix",  # VIX futures (alternative code)
    # G10 FX futures (CME)
    "099741": "eur",  # Euro FX
    "096742": "gbp",  # British Pound
    "097741": "jpy",  # Japanese Yen
    "092741": "chf",  # Swiss Franc
    "090741": "cad",  # Canadian Dollar
    "232741": "aud",  # Australian Dollar
    "112741": "nzd",  # New Zealand Dollar
    "095741": "mxn",  # Mexican Peso (not G10 but heavily traded)
    "089741": "sek",  # Swedish Krona
    "088741": "nok",  # Norwegian Krone
    # US rates futures
    "13874P": "sofr_3m",  # Three-Month SOFR (CME)
    "134741": "ust_10y",  # 10-Year T-Note
    "020601": "ust_30y",  # T-Bond (30Y)
    "044601": "ust_5y",  # 5-Year T-Note
    "042601": "ust_2y",  # 2-Year T-Note
    "043602": "fed_funds",  # 30-Day Federal Funds
}

# Regex fallbacks for market-name-based contract detection.
_MARKET_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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


def _infer_contract_code(cftc_code: str, market_name: str) -> str:
    """Map CFTC contract code or market name to a short identifier."""
    code = cftc_code.strip()
    if code in _CONTRACT_CODES:
        return _CONTRACT_CODES[code]
    for pat, name in _MARKET_NAME_PATTERNS:
        if pat.search(market_name or ""):
            return name
    # Fallback: use the raw CFTC code, cleaned.
    return re.sub(r"[^a-z0-9]", "", code.lower()) or "unknown"


def _publication_date(report_date: pd.Series) -> pd.Series:
    """Compute publication date (Friday) from report date (Tuesday).

    The CFTC publishes COT data on Friday afternoon for positions reported
    as of the preceding Tuesday — a 3 *business-day* lag.  We use
    ``pd.tseries.offsets.BDay(3)`` so that Tuesday → Friday even across
    holidays (it simply rolls forward through any intervening non-business
    days).
    """
    offset = pd.tseries.offsets.BDay(3)
    return report_date.map(lambda d: d + offset if pd.notna(d) else pd.NaT)


class CFTCCoTSource(DataSource):
    """CFTC Commitments of Traders — Traders in Financial Futures (TFF)."""

    name: str = "cftc_cot"
    TABLE = "cftc.cot.tff"
    URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    FIRST_YEAR = 2006

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
            list(trader_categories) if trader_categories else list(_TRADER_CATEGORIES)
        )

    # ── schema ──────────────────────────────────────────────────────────
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

    # ── internal helpers ────────────────────────────────────────────────
    def _year_urls(self, start_year: int, end_year: int) -> list[str]:
        """Build download URLs for the requested year range."""
        if self._file_urls:
            return list(self._file_urls)
        return [
            self._url_template.format(year=y)
            for y in range(max(start_year, self.FIRST_YEAR), end_year + 1)
        ]

    def _read_zip(self, url: str) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=url,
            source="cftc_cot",
            artifact_name=Path(url).stem + ".zip",
        )
        # CFTC TFF ZIPs contain .txt files (CSV-formatted), not .csv,
        # so parse_zip_csv_bytes (which filters for .csv) won't work.
        # Read the first CSV-like file (.csv or .txt) from the archive.
        import io
        import zipfile

        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext in ("csv", "txt"):
                    with zf.open(name) as fh:
                        frame = pd.read_csv(fh)
                    frames.append(normalize_headers(frame))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _to_long(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Melt wide TFF CSV into long format with one row per (date, entity)."""
        if frame.empty:
            return pd.DataFrame()

        # Report date
        date_col = None
        for candidate in [_COL_REPORT_DATE, "report_date", "as_of_date_in_form_yymmdd"]:
            if candidate in frame.columns:
                date_col = candidate
                break
        if date_col is None:
            return pd.DataFrame()

        report_dates = ensure_date_utc(frame[date_col])

        # Contract identification
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
            _infer_contract_code(code, name)
            for code, name in zip(cftc_codes, market_names)
        ]

        # Open interest (shared across categories)
        oi = (
            to_float(frame[_COL_OI])
            if _COL_OI in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )

        rows: list[pd.DataFrame] = []
        for cat_key in self._trader_categories:
            if cat_key not in _TRADER_CATEGORIES:
                continue
            col_long, col_short, col_spread, col_chg_long, col_chg_short, label = (
                _TRADER_CATEGORIES[cat_key]
            )

            # Skip category if columns are missing from the CSV
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
                        if col_spread in frame.columns
                        else float("nan")
                    ),
                    "open_interest": oi,
                    "change_long": (
                        to_float(frame[col_chg_long])
                        if col_chg_long in frame.columns
                        else float("nan")
                    ),
                    "change_short": (
                        to_float(frame[col_chg_short])
                        if col_chg_short in frame.columns
                        else float("nan")
                    ),
                }
            )

            chunk["entity_id"] = [
                make_entity_id("futures", cc, label, "cftc")
                for cc in chunk["contract_code"]
            ]

            rows.append(chunk)

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)
        out["publication_date"] = _publication_date(out["report_date"])
        # Use publication_date as the primary time column for alignment
        out["date"] = out["publication_date"]
        out["asof_utc"] = out["publication_date"]
        return out

    # ── public API ──────────────────────────────────────────────────────
    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        schema = self.schemas()[self.TABLE]

        # Determine year range from query
        start_year = q.start.year if q.start is not None else self.FIRST_YEAR
        end_year = q.end.year if q.end is not None else pd.Timestamp.now(tz="UTC").year

        frames: list[pd.DataFrame] = []
        for url in self._year_urls(start_year, end_year):
            try:
                raw = self._read_zip(url)
            except Exception:
                continue
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
        out = apply_query_filters(out, q=q, time_col="date", entity_col="entity_id")
        out = project_columns(
            out,
            required_columns=schema.required_columns,
            requested_columns=q.columns,
            time_col=schema.time_column,
            entity_col=schema.entity_column,
        )
        return out.reset_index(drop=True)
