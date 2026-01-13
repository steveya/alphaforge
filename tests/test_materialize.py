import pandas as pd
import numpy as np

from alphaforge.features.template import ParamSpec, SliceSpec
from alphaforge.features.realization import FeatureRealization
from alphaforge.features.ops import materialize
from alphaforge.store.cache import MaterializationPolicy
from alphaforge.features.frame import FeatureFrame


class CallCountingTemplate:
    name = "counting"
    version = "1.0"
    param_space = {"p": ParamSpec("int", default=1)}

    def __init__(self):
        self.calls = 0

    def requires(self, params):
        return []

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice, state):
        self.calls += 1
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")], ["AAA"]],
            names=["ts_utc", "entity_id"],
        )
        col = f"f{params['p']}"
        X = pd.DataFrame({col: np.arange(len(idx))}, index=idx)
        catalog = pd.DataFrame([{"feature_id": col, "family": "test"}]).set_index(
            "feature_id"
        )
        return FeatureFrame(X=X, catalog=catalog, meta={"ok": True})


def test_materialize_persist_and_cache(dummy_ctx):
    ctx, _, _ = dummy_ctx
    tmpl = CallCountingTemplate()

    sl = SliceSpec(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-02"),
        entities=["AAA"],
        asof=None,
        grid="sessions:XNYS",
    )
    r = FeatureRealization(
        template=tmpl.name, version=tmpl.version, params={"p": 7}, slice=sl
    )

    # first run persists
    f1 = materialize(ctx, tmpl, r, MaterializationPolicy(persist_mode="always"))
    assert tmpl.calls == 1

    # second run should hit store.get_frame
    f2 = materialize(ctx, tmpl, r, MaterializationPolicy(persist_mode="always"))
    assert tmpl.calls == 1
    assert f1.X.equals(f2.X)
