"""Fetch sample FX, crypto, and index 5-minute series from local First Rate directories."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from alphaforge import (  # noqa: E402
    FirstRateBarsConfig,
    Query,
    build_first_rate_bars_context,
)


def main() -> None:
    # The raw directories in this workspace live two levels above the repo root.
    data_root = REPO_ROOT.parents[1]
    cfg = FirstRateBarsConfig.from_base_dir(data_root)
    ctx = build_first_rate_bars_context(cfg)

    fx = ctx.fetch(
        Query(
            table="fx.contract_price_5m",
            columns=["bar_start_utc", "close", "volume"],
            entities=["AUDUSD"],
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-03T23:59:59Z"),
        )
    ).data.sort_values(["series_key", "obs_date"])

    crypto = ctx.fetch(
        Query(
            table="crypto.contract_price_5m",
            columns=["bar_start_utc", "close", "volume"],
            entities=["BTC"],
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-03T23:59:59Z"),
        )
    ).data.sort_values(["series_key", "obs_date"])

    index = ctx.fetch(
        Query(
            table="index.level_5m",
            columns=["bar_start_utc", "close"],
            entities=["DAX"],
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-03T23:59:59Z"),
        )
    ).data.sort_values(["series_key", "obs_date"])

    print("FX 5-minute bars")
    print(fx.head(10).to_string(index=False))
    print()
    print("Crypto 5-minute bars")
    print(crypto.head(10).to_string(index=False))
    print()
    print("Index 5-minute levels")
    print(index.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
