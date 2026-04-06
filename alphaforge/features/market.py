from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .frame import FeatureFrame
from .ids import group_path, make_feature_id
from .template import ParamSpec, SliceSpec


def _coerce_windows(value: Any, *, name: str, allow_zero: bool = False) -> list[int]:
    if isinstance(value, int):
        items = [int(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [int(item) for item in value]
    else:
        raise TypeError(f"{name} must be an int or sequence of ints.")

    if not items:
        raise ValueError(f"{name} cannot be empty.")

    minimum = 0 if allow_zero else 1
    for item in items:
        if item < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must contain only {qualifier} integers.")
    return list(dict.fromkeys(items))


def _coerce_market_frame(
    ctx,
    *,
    dataset: str,
    source: str | None,
    price_col: str,
    slice: SliceSpec,
) -> pd.DataFrame:
    result = ctx.load(
        dataset,
        columns=[price_col],
        start=slice.start,
        end=slice.end,
        entities=slice.entities,
        asof=slice.asof,
        grid=slice.grid,
        source=source,
    )
    frame = result.data.copy()
    empty_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], tz="UTC"), pd.Index([], dtype="object")],
        names=["ts_utc", "entity_id"],
    )
    if frame.empty:
        return pd.DataFrame({price_col: pd.Series(dtype="float64")}, index=empty_index)

    missing = {"obs_date", "series_key", price_col} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Market template fetch for '{dataset}' is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = frame.loc[:, ["obs_date", "series_key", price_col]].copy()
    frame["ts_utc"] = pd.to_datetime(frame["obs_date"], utc=True)
    frame["entity_id"] = frame["series_key"].astype(str)
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")

    if slice.start is not None:
        frame = frame[frame["ts_utc"] >= pd.Timestamp(slice.start).tz_convert("UTC")]
    if slice.end is not None:
        frame = frame[frame["ts_utc"] <= pd.Timestamp(slice.end).tz_convert("UTC")]
    if slice.asof is not None:
        frame = frame[frame["ts_utc"] <= pd.Timestamp(slice.asof).tz_convert("UTC")]
    if slice.entities is not None:
        allowed = {str(entity) for entity in slice.entities}
        frame = frame[frame["entity_id"].isin(allowed)]

    out = frame.set_index(["ts_utc", "entity_id"])[[price_col]].sort_index()
    return out[~out.index.duplicated(keep="last")]


def _compute_returns(prices: pd.Series, *, return_kind: str) -> pd.Series:
    if return_kind == "log":
        base = np.log(prices.astype(float))
        return base.groupby(level="entity_id").diff()
    if return_kind == "simple":
        return prices.astype(float).groupby(level="entity_id").pct_change()
    raise ValueError("return_kind must be 'log' or 'simple'.")


class LagReturnsTemplate:
    """Lagged return features from a canonical market-price dataset."""

    name = "lag_returns"
    version = "1.0"
    param_space = {
        "lags": ParamSpec("categorical", default=(1, 5, 21)),
        "price_col": ParamSpec("categorical", default="close"),
        "dataset": ParamSpec("categorical", default="market.ohlcv"),
        "source": ParamSpec("categorical", default=None),
        "return_kind": ParamSpec(
            "categorical", default="log", choices=["log", "simple"]
        ),
    }

    def requires(self, params):
        return []

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice: SliceSpec, state):
        dataset = str(params.get("dataset", "market.ohlcv"))
        source = params.get("source")
        price_col = str(params.get("price_col", "close"))
        return_kind = str(params.get("return_kind", "log")).lower()
        lags = _coerce_windows(params.get("lags", (1, 5, 21)), name="lags")

        prices = _coerce_market_frame(
            ctx,
            dataset=dataset,
            source=source,
            price_col=price_col,
            slice=slice,
        )[price_col]
        returns = _compute_returns(prices, return_kind=return_kind)

        features: dict[str, pd.Series] = {}
        catalog_rows: list[dict[str, Any]] = []
        for lag in lags:
            feature_id = make_feature_id(
                dataset,
                "*",
                "market",
                f"{return_kind}_return_lag",
                {"lag": lag, "price_col": price_col},
            )
            features[feature_id] = returns.groupby(level="entity_id").shift(lag)
            catalog_rows.append(
                {
                    "feature_id": feature_id,
                    "group_path": group_path(
                        "market",
                        "lag_returns",
                        {"lags": tuple(lags), "price_col": price_col},
                    ),
                    "family": "market",
                    "transform": "lag_return",
                    "source_table": dataset,
                    "source_name": source,
                    "price_col": price_col,
                    "return_kind": return_kind,
                    "lag": lag,
                }
            )

        X = pd.DataFrame(features, index=prices.index).sort_index()
        catalog = pd.DataFrame(catalog_rows).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X,
            catalog=catalog,
            meta={
                "template": self.name,
                "version": self.version,
                "dataset": dataset,
                "source": source,
            },
        )


class RollingVolatilityTemplate:
    """Rolling realized-volatility features from canonical market-price data."""

    name = "rolling_volatility"
    version = "1.0"
    param_space = {
        "windows": ParamSpec("categorical", default=(5, 21, 63)),
        "lag": ParamSpec("int", default=1, low=0),
        "price_col": ParamSpec("categorical", default="close"),
        "dataset": ParamSpec("categorical", default="market.ohlcv"),
        "source": ParamSpec("categorical", default=None),
        "return_kind": ParamSpec(
            "categorical", default="log", choices=["log", "simple"]
        ),
        "annualization_factor": ParamSpec("int", default=252, low=1),
        "min_periods": ParamSpec("int", default=None, low=1),
    }

    def requires(self, params):
        return []

    def fit(self, ctx, params, fit_slice):
        return None

    def transform(self, ctx, params, slice: SliceSpec, state):
        dataset = str(params.get("dataset", "market.ohlcv"))
        source = params.get("source")
        price_col = str(params.get("price_col", "close"))
        return_kind = str(params.get("return_kind", "log")).lower()
        windows = _coerce_windows(params.get("windows", (5, 21, 63)), name="windows")
        lag = _coerce_windows(params.get("lag", 1), name="lag", allow_zero=True)[0]
        annualization_factor = float(params.get("annualization_factor", 252))
        min_periods_param = params.get("min_periods")

        prices = _coerce_market_frame(
            ctx,
            dataset=dataset,
            source=source,
            price_col=price_col,
            slice=slice,
        )[price_col]
        returns = _compute_returns(prices, return_kind=return_kind)

        features: dict[str, pd.Series] = {}
        catalog_rows: list[dict[str, Any]] = []
        for window in windows:
            min_periods = window if min_periods_param is None else int(min_periods_param)
            realized_vol = returns.groupby(level="entity_id").transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=min_periods,
                ).std()
            )
            if lag:
                realized_vol = realized_vol.groupby(level="entity_id").shift(lag)
            if annualization_factor != 1.0:
                realized_vol = realized_vol * math.sqrt(annualization_factor)

            feature_id = make_feature_id(
                dataset,
                "*",
                "market",
                f"{return_kind}_realized_volatility",
                {
                    "window": window,
                    "lag": lag,
                    "price_col": price_col,
                    "annualization_factor": annualization_factor,
                },
            )
            features[feature_id] = realized_vol
            catalog_rows.append(
                {
                    "feature_id": feature_id,
                    "group_path": group_path(
                        "market",
                        "rolling_volatility",
                        {
                            "window": window,
                            "lag": lag,
                            "price_col": price_col,
                        },
                    ),
                    "family": "market",
                    "transform": "rolling_volatility",
                    "source_table": dataset,
                    "source_name": source,
                    "price_col": price_col,
                    "return_kind": return_kind,
                    "window": window,
                    "lag": lag,
                    "annualization_factor": annualization_factor,
                }
            )

        X = pd.DataFrame(features, index=prices.index).sort_index()
        catalog = pd.DataFrame(catalog_rows).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X,
            catalog=catalog,
            meta={
                "template": self.name,
                "version": self.version,
                "dataset": dataset,
                "source": source,
            },
        )
