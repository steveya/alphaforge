import pandas as pd


def feature_fingerprints(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": X.mean(skipna=True),
            "std": X.std(skipna=True),
            "missing_pct": X.isna().mean(),
        }
    )
