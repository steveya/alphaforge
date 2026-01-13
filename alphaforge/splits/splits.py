from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_splits(
    dates: pd.DatetimeIndex, train_window: int, test_window: int, step: int = 1
) -> Iterator[TimeSplit]:
    dates = pd.DatetimeIndex(dates).sort_values()
    n = len(dates)
    for i in range(train_window, n - test_window + 1, step):
        yield TimeSplit(
            dates[i - train_window], dates[i - 1], dates[i], dates[i + test_window - 1]
        )


def expanding_splits(
    dates: pd.DatetimeIndex, min_train: int, test_window: int, step: int = 1
) -> Iterator[TimeSplit]:
    dates = pd.DatetimeIndex(dates).sort_values()
    n = len(dates)
    for i in range(min_train, n - test_window + 1, step):
        yield TimeSplit(dates[0], dates[i - 1], dates[i], dates[i + test_window - 1])


def purged_kfold_splits(
    dates: pd.DatetimeIndex, n_splits: int = 5, embargo: int = 0
) -> list[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Purged time-series k-fold on date index with simple embargo in business days."""
    dates = pd.DatetimeIndex(dates).sort_values()
    n = len(dates)
    fold = n // n_splits
    out = []
    for k in range(n_splits):
        a = k * fold
        b = (k + 1) * fold if k < n_splits - 1 else n
        test = dates[a:b]
        if len(test) == 0:
            continue

        ts0, ts1 = test.min(), test.max()
        if embargo > 0:
            emb0 = ts0 - pd.tseries.offsets.BDay(embargo)
            emb1 = ts1 + pd.tseries.offsets.BDay(embargo)
            train = dates[(dates < emb0) | (dates > emb1)]
        else:
            train = dates.difference(test)

        out.append((train, test))
    return out
