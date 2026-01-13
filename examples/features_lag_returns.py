import numpy as np
import pandas as pd
from alphaforge.data.query import Query
from alphaforge.features.template import ParamSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path


class LagReturnsTemplate:
    name = "lag_returns"
    version = "1.0"
    param_space = {
        "lags": ParamSpec("int", default=5, low=1, high=252),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
        "source": ParamSpec("categorical", default="dummy", choices=["dummy"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
    }

    def requires(self, params):
        return []

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice, state):
        lags = int(params["lags"])
        source = params["source"]
        table = params["table"]
        price_col = params["price_col"]

        panel = ctx.fetch_panel(
            source,
            Query(
                table=table,
                columns=[price_col],
                start=slice.start,
                end=slice.end,
                entities=slice.entities,
                asof=slice.asof,
                grid=slice.grid,
            ),
        )
        df = panel.df.copy()
        px = df[price_col].astype(float)
        logret = np.log(px).groupby(level="entity_id").diff()

        X_cols = {}
        cat = []
        for k in range(1, lags + 1):
            fid = make_feature_id(table, "*", "lag", "logret", {"k": k})
            X_cols[fid] = logret.groupby(level="entity_id").shift(k)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path("lag", "logret", {"lags": lags}),
                    "family": "lag",
                    "transform": "logret",
                    "source_table": table,
                    "lag": k,
                }
            )
        X = pd.DataFrame(X_cols, index=df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )
