import pandas as pd
import numpy as np
from ..data.panel import PanelFrame


def data_quality_report(panel: PanelFrame) -> pd.DataFrame:
    df = panel.df
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(
            {
                "field": c,
                "n": int(len(s)),
                "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "mean": (
                    float(s.mean(skipna=True))
                    if pd.api.types.is_numeric_dtype(s)
                    else np.nan
                ),
                "std": (
                    float(s.std(skipna=True))
                    if pd.api.types.is_numeric_dtype(s)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("field")


def feature_quality_report(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_pct": X.isna().mean(),
            "mean": X.mean(skipna=True),
            "std": X.std(skipna=True),
        }
    )
