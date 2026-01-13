import pandas as pd
from alphaforge.data.query import Query
from alphaforge.time.grids import SessionGrid
from alphaforge.time.align import AlignSpec, align_panel
from alphaforge.features.template import ParamSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path


class MacroCarryTemplate:
    """
    Demonstrates:
      - low-freq series aligned to session grid
      - ffill (carry) + missingness indicator (structural)
    """

    name = "macro_carry"
    version = "1.0"
    param_space = {
        "source": ParamSpec("categorical", default="dummy", choices=["dummy"]),
        "table": ParamSpec(
            "categorical", default="macro.series", choices=["macro.series"]
        ),
        "value_col": ParamSpec("categorical", default="value", choices=["value"]),
        "method": ParamSpec("categorical", default="ffill", choices=["ffill", "none"]),
    }

    def requires(self, params):
        return []

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice, state):
        source, table, col, method = (
            params["source"],
            params["table"],
            params["value_col"],
            params["method"],
        )

        panel = ctx.fetch_panel(
            source,
            Query(
                table=table,
                columns=[col],
                start=slice.start,
                end=slice.end,
                entities=None,  # macro entity key (e.g., CPI) is internal to table
                asof=slice.asof,
                grid=slice.grid,
            ),
        )
        schema = ctx.sources[source].schemas()[table]

        # build a session grid from calendar
        cal = ctx.calendars["XNYS"]
        idx = cal.sessions(str(slice.start.date()), str(slice.end.date())).tz_convert(
            None
        )
        grid = SessionGrid(name="sessions:XNYS", index=idx, calendar="XNYS")

        aligned = align_panel(
            panel, schema, grid, AlignSpec(target_grid="sessions:XNYS", method=method)
        )

        # Value and a structural-missingness indicator as features
        val = aligned.value.df[[col]].copy()
        avail = aligned.availability.df[[col]].copy()

        fid_val = make_feature_id(
            table, "*", "macro", "carry", {"col": col, "method": method}
        )
        fid_miss = make_feature_id(
            table, "*", "macro", "missing_structural", {"col": col}
        )

        X = pd.DataFrame(index=val.index)
        X[fid_val] = val[col]
        X[fid_miss] = (
            avail[col].isin(["no_update_expected", "not_yet_released"]).astype(float)
        )

        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid_val,
                    "group_path": group_path("macro", "carry", {"col": col}),
                    "family": "macro",
                    "transform": "carry",
                    "source_table": table,
                },
                {
                    "feature_id": fid_miss,
                    "group_path": group_path(
                        "macro", "missing_structural", {"col": col}
                    ),
                    "family": "macro",
                    "transform": "missing_structural",
                    "source_table": table,
                },
            ]
        ).set_index("feature_id")

        return FeatureFrame(
            X=X.sort_index(),
            catalog=catalog.sort_index(),
            meta={"template": self.name, "version": self.version},
        )
