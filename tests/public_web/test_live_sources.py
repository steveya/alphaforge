from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.public_web import (
    CFTCWeeklySwapsSource,
    CMEProductSlateSource,
    DTCCPPDSource,
    EurexRefdataContractsSource,
    EurexStatsDailySource,
    EzoicAdRevenueDailySource,
    LCHCDSClearDailySource,
)
from alphaforge.data.public_web.http import CachedHttpClient
from alphaforge.data.query import Query

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


RUN_NETWORK = os.getenv("ALPHAFORGE_NETWORK_TESTS") == "1"


def _build_test_http_client() -> CachedHttpClient:
    root = Path(__file__).resolve().parents[2]
    base = os.getenv("ALPHAFORGE_TEST_CACHE_DIR", str(root / ".cache/public_web_live"))
    tag = os.getenv("ALPHAFORGE_TEST_CACHE_TAG", "v1")
    cache_dir = Path(base) / tag
    return CachedHttpClient(cache_dir=cache_dir, cache_partition="static")


def _fetch_or_skip(fetch_fn, context: str):
    try:
        df = fetch_fn()
        if getattr(df, "empty", False):
            pytest.skip(f"Skipping {context} because live response was empty")
        return df
    except Exception as exc:  # pragma: no cover - network/provider variability
        pytest.skip(f"Skipping {context} due to live-source error: {exc}")


def _call_or_skip(call_fn, context: str):
    try:
        return call_fn()
    except Exception as exc:  # pragma: no cover - network/provider variability
        pytest.skip(f"Skipping {context} due to live-source error: {exc}")


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_cme_productslate_smoke() -> None:
    src = CMEProductSlateSource(http_client=_build_test_http_client())
    df = _fetch_or_skip(
        lambda: src.fetch(
            Query(
                table="cme.productslate.reference",
                columns=["product_name"],
            )
        ),
        "cme product slate",
    )
    assert not df.empty
    assert {"date", "entity_id", "asof_utc"}.issubset(df.columns)


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_cftc_swaps_smoke() -> None:
    src = CFTCWeeklySwapsSource(http_client=_build_test_http_client())
    df = _fetch_or_skip(
        lambda: src.fetch(
            Query(
                table="cftc.swaps.weekly",
                columns=["value"],
                start=pd.Timestamp("2024-01-01", tz="UTC"),
            )
        ),
        "cftc swaps weekly",
    )
    assert not df.empty


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_eurex_stats_smoke() -> None:
    src = EurexStatsDailySource(http_client=_build_test_http_client())
    df = _fetch_or_skip(
        lambda: src.fetch(
            Query(table="eurex.stats.daily", columns=["volume", "open_interest"])
        ),
        "eurex stats",
    )
    assert not df.empty


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_lch_cdsclear_smoke() -> None:
    src = LCHCDSClearDailySource(http_client=_build_test_http_client())
    df = _fetch_or_skip(
        lambda: src.fetch(Query(table="lch.cdsclear.daily", columns=["value"])),
        "lch cdsclear",
    )
    assert not df.empty


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_ezoic_smoke() -> None:
    src = EzoicAdRevenueDailySource(http_client=_build_test_http_client())
    df = _fetch_or_skip(
        lambda: src.fetch(Query(table="ezoic.adrevenue.daily", columns=["value"])),
        "ezoic adrevenue",
    )
    assert not df.empty


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_dtcc_smoke() -> None:
    src = DTCCPPDSource(
        http_client=_build_test_http_client(),
    )
    df = _fetch_or_skip(
        lambda: src.fetch(
            Query(
                table="dtcc.ppd.events",
                columns=["notional", "price"],
                start=pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=1),
                end=pd.Timestamp.utcnow().tz_convert("UTC"),
            )
        ),
        "dtcc ppd",
    )
    assert {"ts_utc", "entity_id", "asof_utc"}.issubset(df.columns)


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_dtcc_list_endpoint_schema_smoke() -> None:
    src = DTCCPPDSource(http_client=_build_test_http_client())
    rows = _call_or_skip(
        lambda: src._list_reports("cumulative", "IR"),
        "dtcc cumulative list endpoint",
    )
    if not rows:
        pytest.skip("Skipping dtcc cumulative list endpoint because response was empty")

    first = rows[0]
    assert "fileName" in first
    assert str(first["fileName"]).endswith(".zip")
    assert "dissemDTM" in first


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="Set ALPHAFORGE_NETWORK_TESTS=1 to run live source tests"
)
def test_live_eurex_refdata_if_api_set() -> None:
    api_url = os.getenv("EUREX_REFDATA_API_URL")
    if not api_url:
        pytest.skip("Set EUREX_REFDATA_API_URL to run Eurex refdata live test")

    src = EurexRefdataContractsSource(
        api_url=api_url,
        http_client=_build_test_http_client(),
    )
    df = _fetch_or_skip(
        lambda: src.fetch(
            Query(table="eurex.refdata.contracts", columns=["symbol", "currency"])
        ),
        "eurex refdata",
    )
    assert not df.empty
