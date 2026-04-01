from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge import FirstRateBarsConfig, Query, build_first_rate_bars_context


def _write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_first_rate_bars_context_discovers_entities_and_fetches_rows(tmp_path) -> None:
    base_dir = tmp_path / "raw"
    fx_dir = base_dir / "fx_contract_price_5m"
    crypto_dir = base_dir / "crypto_contract_price_5m"
    index_dir = base_dir / "index_level_5m"
    fx_dir.mkdir(parents=True)
    crypto_dir.mkdir()
    index_dir.mkdir()

    _write_rows(
        fx_dir / "AUDUSD_full_5min.txt",
        [
            "20100103,17:00:00,0.8981,0.89824,0.89807,0.89824,14",
            "20100103,17:05:00,0.89824,0.89839,0.89824,0.89839,30",
        ],
    )
    _write_rows(
        crypto_dir / "BTC_full_5min.txt",
        [
            "2013-04-01 00:30:00,93.183,93.183,93.183,93.183,13.02",
            "2013-04-01 00:35:00,93.200,93.250,93.100,93.240,20.0",
        ],
    )
    _write_rows(
        index_dir / "DAX_full_5min.txt",
        [
            "2008-01-02 03:00:00,8038.04,8055.72,8027.91,8051.81",
            "2008-01-02 03:05:00,8051.80,8062.14,8050.92,8061.44",
        ],
    )

    cfg = FirstRateBarsConfig.from_base_dir(base_dir)
    ctx = build_first_rate_bars_context(cfg)
    adapter = ctx.adapters["first_rate_bars"]

    assert adapter.list_entities("fx.contract_price_5m") == ["AUDUSD"]
    assert adapter.list_entities("crypto.contract_price_5m") == ["BTC"]
    assert adapter.list_entities("index.level_5m") == ["DAX"]

    fx = ctx.fetch(
        Query(
            table="fx.contract_price_5m",
            columns=["bar_start_utc", "close", "volume"],
            entities=["AUDUSD"],
            start="2010-01-03T22:05:00Z",
            end="2010-01-03T22:10:00Z",
        )
    )
    assert fx.source == "first_rate_bars"
    assert fx.dataset == "fx.contract_price_5m"
    assert fx.is_pit is False
    assert fx.data["series_key"].tolist() == ["AUDUSD", "AUDUSD"]
    assert fx.data["obs_date"].tolist() == [
        pd.Timestamp("2010-01-03 22:05:00+00:00"),
        pd.Timestamp("2010-01-03 22:10:00+00:00"),
    ]
    assert fx.data["bar_start_utc"].tolist()[0] == pd.Timestamp("2010-01-03 22:00:00+00:00")

    crypto = ctx.fetch(
        Query(
            table="crypto.contract_price_5m",
            columns=["close", "volume"],
            entities=["BTC"],
        )
    )
    assert crypto.data["series_key"].tolist() == ["BTC", "BTC"]
    assert crypto.data["close"].tolist() == [93.183, 93.24]

    index = ctx.fetch(
        Query(
            table="index.level_5m",
            columns=["bar_start_utc", "close"],
            entities=["DAX"],
        )
    )
    assert index.data["series_key"].tolist() == ["DAX", "DAX"]
    assert "volume" not in index.data.columns
    assert index.data["bar_start_utc"].tolist()[0] == pd.Timestamp("2008-01-02 08:00:00+00:00")
