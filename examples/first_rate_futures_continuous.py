"""Fetch WTI and Brent continuous futures series from local FRD artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphaforge import (  # noqa: E402
    FirstRateFuturesConfig,
    FirstRateFuturesLoader,
    Query,
    build_first_rate_futures_context,
)


def main() -> None:
    cfg = FirstRateFuturesConfig.from_env()

    # Ingest raw contract files into canonical parquet artifacts if needed.
    loader = FirstRateFuturesLoader(cfg)
    loader.ingest()

    ctx = build_first_rate_futures_context(cfg)

    eod = ctx.fetch(
        Query(
            table="futures.continuous_eod_research",
            columns=["open", "high", "low", "close", "volume", "active_contract_id"],
            entities=["CL", "BZ"],  # WTI and Brent
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-03-31T23:59:59Z"),
        )
    ).data.sort_values(["series_key", "obs_date"])

    intraday = ctx.fetch(
        Query(
            table="futures.continuous_5m_execution",
            columns=["close", "volume", "active_contract_id", "roll_flag"],
            entities=["CL", "BZ"],
            start=pd.Timestamp("2024-03-01T00:00:00Z"),
            end=pd.Timestamp("2024-03-05T23:59:59Z"),
        )
    ).data.sort_values(["series_key", "obs_date"])

    print("Continuous EOD research series")
    print(eod.head(10).to_string(index=False))
    print()
    print("Continuous 5-minute execution series")
    print(intraday.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
