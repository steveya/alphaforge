import pandas as pd
import numpy as np

from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext
from alphaforge.store.local_parquet import LocalParquetStore
from alphaforge.features.template import SliceSpec
from alphaforge.features.realization import FeatureRealization
from alphaforge.store.cache import MaterializationPolicy
from alphaforge.features.ops import materialize, join_feature_frames
from alphaforge.splits.splits import rolling_splits

from examples.dummy_source import DummySource
from examples.features_lag_returns import LagReturnsTemplate
from examples.features_macro_carry import MacroCarryTemplate


def main():
    cal = TradingCalendar("XNYS", tz="UTC")
    # sessions returned as tz-aware UTC timestamps
    dates = cal.sessions("2020-01-01", "2020-03-31")
    entities = ["AAA", "BBB"]

    # daily closes
    rng = np.random.default_rng(123)
    rows = []
    for e in entities:
        px = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
        for d, p in zip(dates, px):
            rows.append({"date": d, "entity_id": e, "close": float(p)})
    ohlcv = pd.DataFrame(rows)

    # monthly macro points
    macro = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-31"), "entity_id": "CPI", "value": 1.0},
            {"date": pd.Timestamp("2020-02-29"), "entity_id": "CPI", "value": 2.0},
            {"date": pd.Timestamp("2020-03-31"), "entity_id": "CPI", "value": 3.0},
        ]
    )

    src = DummySource(ohlcv, macro)
    store = LocalParquetStore("alphaforge_demo_store")
    ctx = DataContext(sources={"dummy": src}, calendars={"XNYS": cal}, store=store)

    sl = SliceSpec(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-03-31"),
        entities=entities,
        asof=None,
        grid="sessions:XNYS",
    )

    lag_t = LagReturnsTemplate()
    mac_t = MacroCarryTemplate()

    lag_r = FeatureRealization(
        "lag_returns",
        "1.0",
        {"lags": 5, "source": "dummy", "table": "market.ohlcv", "price_col": "close"},
        sl,
    )
    mac_r = FeatureRealization(
        "macro_carry",
        "1.0",
        {
            "source": "dummy",
            "table": "macro.series",
            "value_col": "value",
            "method": "ffill",
        },
        sl,
    )

    f_lag = materialize(ctx, lag_t, lag_r, MaterializationPolicy(persist_mode="always"))
    f_mac = materialize(ctx, mac_t, mac_r, MaterializationPolicy(persist_mode="always"))

    joined = join_feature_frames([f_lag, f_mac], join="inner")

    print("Joined X shape:", joined.X.shape)
    print("Catalog head:")
    print(joined.catalog.head(10))

    # show a couple splits
    uniq_dates = joined.X.index.get_level_values("ts_utc").unique().sort_values()
    splits = list(rolling_splits(uniq_dates, train_window=20, test_window=5, step=10))
    print("Example splits:", splits[:2])


if __name__ == "__main__":
    main()
