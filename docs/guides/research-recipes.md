# Research Recipes

This guide records notebook-shaped patterns that Alphaforge should support
without bespoke helper layers.

## Volatility recipe

The canonical volatility workflow is:

1. build a `DataContext` from adapters with `DataContext.from_adapters(...)`
2. group reusable features with `FeatureRequestGroup`
3. use built-in market templates for lagged returns and trailing volatility
4. keep custom code focused on the target definition rather than feature
   plumbing

The built-in feature family now covers the common market-price pieces:

- `LagReturnsTemplate` for lagged simple or log returns
- `RollingVolatilityTemplate` for trailing realized volatility windows

Example feature group:

```python
from alphaforge.features import LagReturnsTemplate, RollingVolatilityTemplate
from alphaforge.features.dataset_spec import FeatureRequest, FeatureRequestGroup

features = [
    FeatureRequestGroup(
        key="volatility",
        tags={"recipe": "volatility"},
        requests=[
            FeatureRequest(
                template=LagReturnsTemplate(),
                key="returns",
                params={
                    "dataset": "market.ohlcv",
                    "source": "market",
                    "price_col": "close",
                    "lags": [1, 5, 10],
                },
            ),
            FeatureRequest(
                template=RollingVolatilityTemplate(),
                key="trailing_vol",
                params={
                    "dataset": "market.ohlcv",
                    "source": "market",
                    "price_col": "close",
                    "windows": [5, 10, 21],
                    "lag": 1,
                    "annualization_factor": 252,
                },
            ),
        ],
    )
]
```

This removes the repeated notebook work of:

- fetching prices manually for each feature family
- writing ad hoc lag-return helper cells
- rebuilding the trailing-volatility calculation in each notebook
- hand-annotating request metadata in the catalog

For a full runnable example, see `examples/volatility_dataset_recipe.py`.
