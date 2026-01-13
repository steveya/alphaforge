import pandas as pd
from alphaforge.features.realization import FeatureRealization
from alphaforge.features.template import SliceSpec


def test_realization_id_stability_and_change():
    sl = SliceSpec(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-31"),
        entities=["AAA"],
        asof=None,
        grid="sessions:XNYS",
    )
    r1 = FeatureRealization(
        template="lag_returns", version="1.0", params={"lags": 5}, slice=sl
    )
    r2 = FeatureRealization(
        template="lag_returns", version="1.0", params={"lags": 5}, slice=sl
    )
    assert r1.id() == r2.id()

    r3 = FeatureRealization(
        template="lag_returns", version="1.0", params={"lags": 6}, slice=sl
    )
    assert r1.id() != r3.id()
