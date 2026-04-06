"""MOF Japan Government Bond (JGB) constant-maturity yield curve data source.

Fetches daily par-yield data from the Ministry of Finance (MOF) Japan website
and exposes it via the standard alphaforge ``DataSource`` interface.

Usage
-----
>>> from alphaforge.data.public_web.mof_jgb import MOFJGBYieldCurveSource
>>> from alphaforge.data.query import Query
>>> src = MOFJGBYieldCurveSource()
>>> q = Query(table="mof.jgb.yields", columns=["yield_pct"])
>>> df = src.fetch(q)           # long-form DataFrame
>>> wide = src.fetch_wide(q)    # convenience: date × tenor pivot
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .http import CachedHttpClient
from .schema_helpers import table_schema
from .utils import (
    ensure_date_utc,
    make_entity_id,
)


class MOFJGBYieldCurveSource(PublicWebSourceBase):
    """Daily JGB constant-maturity par yields from the MOF website."""

    name: str = "mof_jgb_yields"
    TABLE = "mof.jgb.yields"

    LANDING_URL = (
        "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm"
    )
    CURRENT_CSV_URL = (
        "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
    )

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        csv_url: str | None = None,
        landing_url: str | None = None,
    ) -> None:
        super().__init__(http_client=http_client, cache_dir=cache_dir)
        self._csv_url = csv_url or self.CURRENT_CSV_URL
        self._landing_url = landing_url or self.LANDING_URL

    # ---- schema --------------------------------------------------------------

    def schemas(self):
        return {
            self.TABLE: table_schema(
                self.TABLE,
                required_columns=["yield_pct"],
                canonical_columns=[
                    "yield_pct",
                    "tenor",
                    "maturity_years",
                ],
                native_freq="B",
                time_semantics="point",
            )
        }

    # ---- discovery -----------------------------------------------------------

    def _discover_csv_urls(self) -> list[str]:
        """Find CSV download links from the MOF landing page."""
        try:
            payload = self._http.get_bytes(
                url=self._landing_url,
                source="mof_jgb",
                artifact_name="landing.html",
            )
        except Exception:
            return [self._csv_url]

        html = payload.decode(errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)

        scored: list[tuple[int, str]] = []
        for h in hrefs:
            if not h:
                continue
            u = urljoin(self._landing_url, h)
            low = u.lower()
            if ".csv" not in low:
                continue
            s = 0
            if "jgb" in low:
                s += 4
            if "constant" in low or "cme" in low:
                s += 4
            if "historical" in low or "all" in low or "archive" in low:
                s += 2
            scored.append((s, u))

        scored.sort(key=lambda x: (-x[0], x[1]))
        urls: list[str] = list(dict.fromkeys(u for _, u in scored))
        if self._csv_url not in urls:
            urls.append(self._csv_url)
        return urls

    # ---- CSV parsing ---------------------------------------------------------

    @staticmethod
    def _decode_bytes(data: bytes) -> str:
        for enc in ("utf-8-sig", "cp932", "shift_jis", "utf-8"):
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return data.decode(errors="replace")

    @staticmethod
    def _coerce_float(text: str) -> float:
        token = str(text).strip().replace(",", "").replace("%", "")
        if token in {"", "-", "--", "NA", "N/A", "na", "n/a"}:
            return np.nan
        try:
            return float(token)
        except (ValueError, TypeError):
            return np.nan

    @staticmethod
    def _parse_maturity_years(label: str) -> float | None:
        token = str(label).strip().upper().replace(" ", "")
        token = (
            token.replace("YEARS", "Y")
            .replace("YEAR", "Y")
            .replace("YRS", "Y")
            .replace("YR", "Y")
        )
        token = token.replace("MONTHS", "M").replace("MONTH", "M").replace("MON", "M")
        ym = re.search(r"(\d+(?:\.\d+)?)Y", token)
        if ym:
            return float(ym.group(1))
        mm = re.search(r"(\d+(?:\.\d+)?)M", token)
        if mm:
            return float(mm.group(1)) / 12.0
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return float(token)
        return None

    def _parse_csv(self, data: bytes) -> pd.DataFrame:
        """Parse a MOF-style CSV into a wide (date × tenor) DataFrame."""
        text = self._decode_bytes(data)
        rows = list(csv.reader(io.StringIO(text)))
        rows = [r for r in rows if any(str(c).strip() for c in r)]

        # Detect header row
        header_idx: int | None = None
        date_col = 0
        for i, row in enumerate(rows[:60]):
            stripped = [c.strip() for c in row]
            date_cols = [
                j for j, v in enumerate(stripped) if re.search("date", v, flags=re.I)
            ]
            m_hits = sum(self._parse_maturity_years(v) is not None for v in stripped)
            if date_cols and m_hits >= 3:
                header_idx, date_col = i, date_cols[0]
                break
        if header_idx is None:
            for i, row in enumerate(rows[:60]):
                m_hits = sum(
                    self._parse_maturity_years(v.strip()) is not None for v in row
                )
                if m_hits >= 4:
                    header_idx, date_col = i, 0
                    break
        if header_idx is None:
            raise ValueError("Unable to detect MOF CSV header row")

        header = rows[header_idx]
        tenor_info: list[tuple[int, float, str]] = []
        for j, label in enumerate(header):
            if j == date_col:
                continue
            years = self._parse_maturity_years(label)
            if years is not None:
                tenor_info.append((j, years, str(label).strip()))

        recs: list[dict] = []
        for row in rows[header_idx + 1 :]:
            if len(row) <= date_col:
                continue
            dt = pd.to_datetime(str(row[date_col]).strip(), errors="coerce")
            if pd.isna(dt):
                continue
            rec: dict = {"Date": dt}
            for j, _years, label in tenor_info:
                token = row[j] if j < len(row) else ""
                rec[label] = self._coerce_float(token)
            recs.append(rec)

        if not recs:
            raise ValueError("No valid data rows in MOF CSV")

        df = pd.DataFrame(recs).drop_duplicates("Date").set_index("Date").sort_index()

        # Normalise column names
        rename_map: dict[str, str] = {}
        for _j, years, label in tenor_info:
            if years >= 1:
                rename_map[label] = f"{years:g}Y"
            else:
                rename_map[label] = f"{int(round(years * 12)):d}M"
        df = df.rename(columns=rename_map)
        df = df.T.groupby(level=0).mean().T
        ordered = sorted(
            df.columns,
            key=lambda c: self._parse_maturity_years(c) or np.inf,
        )
        return df[ordered].astype(float)

    # ---- long-form conversion ------------------------------------------------

    def _to_long(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Pivot wide (date × tenor) yield table to alphaforge long form."""
        records: list[dict] = []
        for dt in wide.index:
            for tenor in wide.columns:
                val = wide.at[dt, tenor]
                if pd.isna(val):
                    continue
                mat_years = self._parse_maturity_years(tenor)
                records.append(
                    {
                        "date": pd.Timestamp(dt),
                        "tenor": tenor,
                        "maturity_years": mat_years,
                        "yield_pct": float(val),
                        "entity_id": make_entity_id(
                            "rates", "jgb", "yield", tenor.lower()
                        ),
                    }
                )
        out = pd.DataFrame(records)
        if not out.empty:
            out["date"] = ensure_date_utc(out["date"])
            out["asof_utc"] = out["date"]
        return out

    # ---- fetch ---------------------------------------------------------------

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)

        schema = self._schema()

        frames: list[pd.DataFrame] = []
        for url in self._discover_csv_urls():
            try:
                payload = self._http.get_bytes(
                    url=url,
                    source="mof_jgb",
                    artifact_name=Path(url).name or "jgb_yields.csv",
                )
                wide = self._parse_csv(payload)
                long = self._to_long(wide)
                if not long.empty:
                    frames.append(long)
                    break  # first successful parse is sufficient
            except Exception:
                continue

        if not frames:
            return self._empty_frame(schema)

        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, q=q, schema=schema, sort_by=[])

    # ---- convenience ---------------------------------------------------------

    def fetch_wide(self, q: Query | None = None) -> pd.DataFrame:
        """Return a date × tenor DataFrame (yields in percent).

        This is the natural format for yield-curve analysis.
        """
        if q is None:
            q = Query(table=self.TABLE, columns=["yield_pct", "tenor"])
        long = self.fetch(q)
        if long.empty:
            return pd.DataFrame()
        return long.pivot_table(
            index="date", columns="tenor", values="yield_pct"
        ).sort_index()
