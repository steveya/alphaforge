# Evaluation Metrics

Pluggable forecast accuracy metrics for composing evaluation pipelines.

The [`MetricFn`][alphaforge.evaluation.metrics.MetricFn] protocol defines
the interface that all metrics must satisfy.  Five built-in implementations
are provided, and two convenience suites (`DEFAULT_METRICS` and
`BENCHMARK_METRICS`) bundle the most commonly used combinations.

## Quick start

```python
from alphaforge.evaluation.metrics import RMSE, MAE, BENCHMARK_METRICS

# Single metric
rmse = RMSE()
score = rmse(y_pred, y_true)

# Full benchmark suite
for metric in BENCHMARK_METRICS:
    print(f"{metric.name}: {metric(y_pred, y_true):.4f}")
```

## Custom metrics

Any class with a `name` attribute and `__call__(y_pred, y_true) -> float`
satisfies the protocol:

```python
import numpy as np
from alphaforge.evaluation.metrics import MetricFn

class MedianAbsoluteError:
    name = "median_ae"

    def __call__(self, y_pred, y_true):
        return float(np.median(np.abs(y_pred - y_true)))

assert isinstance(MedianAbsoluteError(), MetricFn)
```

## API Reference

::: alphaforge.evaluation.metrics
