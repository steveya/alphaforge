from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import pandas as pd

_TENOR_BUCKETS_YEARS = [
    (1 / 52, "1w"),
    (2 / 52, "2w"),
    (1 / 12, "1m"),
    (2 / 12, "2m"),
    (3 / 12, "3m"),
    (6 / 12, "6m"),
    (9 / 12, "9m"),
    (1.0, "1y"),
    (2.0, "2y"),
    (3.0, "3y"),
    (4.0, "4y"),
    (5.0, "5y"),
    (6.0, "6y"),
    (7.0, "7y"),
    (8.0, "8y"),
    (9.0, "9y"),
    (10.0, "10y"),
    (12.0, "12y"),
    (15.0, "15y"),
    (20.0, "20y"),
    (25.0, "25y"),
    (30.0, "30y"),
    (40.0, "40y"),
    (50.0, "50y"),
]

_CCY_MAP = {
    "usd": "usd",
    "us dollar": "usd",
    "dollar": "usd",
    "eur": "eur",
    "euro": "eur",
    "jpy": "jpy",
    "yen": "jpy",
    "gbp": "gbp",
    "sterling": "gbp",
    "chf": "chf",
    "cad": "cad",
    "aud": "aud",
    "nzd": "nzd",
    "sek": "sek",
    "nok": "nok",
}


def snake_case(name: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    text = re.sub(r"_+", "_", text)
    return text.lower()


def make_entity_id(*parts: str) -> str:
    cleaned: list[str] = []
    for part in parts:
        val = (part or "").strip().lower().replace(" ", "_")
        val = re.sub(r"[^a-z0-9_.-]", "", val)
        val = re.sub(r"\.+", ".", val).strip(".")
        if val:
            cleaned.append(val)
    return ".".join(cleaned)


def normalize_ccy(s: str | None) -> str | None:
    if s is None:
        return None
    key = s.strip().lower()
    if not key:
        return None
    return _CCY_MAP.get(key, key[:3] if len(key) >= 3 else None)


def normalize_pair(base: str, quote: str) -> str:
    base_norm = normalize_ccy(base) or "unk"
    quote_norm = normalize_ccy(quote) or "unk"
    return f"{base_norm}{quote_norm}"


def bucket_tenor(raw: str | float | pd.Timedelta | None) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "unk"

    years: float | None = None

    if isinstance(raw, pd.Timedelta):
        years = raw / pd.Timedelta(days=365.25)
    elif isinstance(raw, (int, float)):
        years = float(raw)
    elif isinstance(raw, str):
        text = raw.strip().lower()
        match = re.match(r"^(\d+(?:\.\d+)?)\s*([dwmy])$", text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "d":
                years = value / 365.25
            elif unit == "w":
                years = value / 52.0
            elif unit == "m":
                years = value / 12.0
            elif unit == "y":
                years = value
        else:
            tenor = text.replace(" ", "")
            if tenor in {bucket for _, bucket in _TENOR_BUCKETS_YEARS}:
                return tenor

    if years is None:
        return "unk"

    best = min(_TENOR_BUCKETS_YEARS, key=lambda x: abs(x[0] - years))
    return best[1]


def ensure_utc(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if getattr(values.dt, "tz", None) is None:
        return values.dt.tz_localize("UTC")
    return values.dt.tz_convert("UTC")


def ensure_date_utc(series: pd.Series) -> pd.Series:
    out = ensure_utc(series)
    normalized = out.map(lambda ts: ts.normalize() if pd.notna(ts) else pd.NaT)
    return pd.to_datetime(normalized, utc=True)


def coalesce_columns(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([None] * len(df), index=df.index)


def to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def apply_query_filters(
    df: pd.DataFrame,
    *,
    q: Any,
    time_col: str,
    entity_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if q.start is not None:
        out = out[out[time_col] >= q.start]
    if q.end is not None:
        out = out[out[time_col] <= q.end]
    if q.entities is not None:
        requested = {str(x) for x in q.entities}
        out = out[out[entity_col].astype(str).isin(requested)]
    if getattr(q, "asof", None) is not None and "asof_utc" in out.columns:
        out = out[out["asof_utc"] <= q.asof]
    return out


def project_columns(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str],
    requested_columns: Iterable[str],
    time_col: str,
    entity_col: str,
) -> pd.DataFrame:
    columns = [time_col, entity_col, "asof_utc"]
    for col in list(required_columns) + list(requested_columns):
        if col in df.columns and col not in columns:
            columns.append(col)
    return df[columns].copy()


def first_existing(df: pd.DataFrame, *names: str) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None
