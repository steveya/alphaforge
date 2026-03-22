"""Pluggable forecast accuracy metrics.

This module defines a :class:`MetricFn` protocol and a library of standard
implementations that can be composed into any evaluation pipeline.  The
protocol is ``runtime_checkable``, so ``isinstance(obj, MetricFn)`` works
for custom metrics without inheriting from a base class.

Design rationale
~~~~~~~~~~~~~~~~

Published nowcasting benchmarks report different metric sets — GDPNow and
the Atlanta Fed use RMSE/MAE, the NY Fed reports Log Predictive Score, the
IMF reports Directional Accuracy, and the ECB uses CRPS.  Rather than
hard-coding metric logic into evaluation functions, we define a thin protocol
and let callers compose their own metric suites.

Creating a custom metric
~~~~~~~~~~~~~~~~~~~~~~~~

Any class satisfying the :class:`MetricFn` protocol works.  The only
requirements are a ``name`` property (used as the column header in result
DataFrames) and a ``__call__(y_pred, y_true) -> float`` method::

    class MedianAbsoluteError:
        name = "median_ae"

        def __call__(self, y_pred, y_true):
            return float(np.median(np.abs(y_pred - y_true)))

Then pass it to any evaluation function::

    from alphaforge.evaluation.metrics import RMSE
    decompose_accuracy_by_horizon(predictions, metrics=[RMSE(), MedianAbsoluteError()])

Pre-built suites
~~~~~~~~~~~~~~~~

Two convenience tuples are provided:

- :data:`DEFAULT_METRICS` — ``(RMSE, MAE, DirectionalAccuracy)`` for general
  use where only basic accuracy is needed.
- :data:`BENCHMARK_METRICS` — adds ``MeanError`` (bias) and ``MAPE`` for
  benchmark comparison tables that need to match published reporting.

Examples
--------
>>> import numpy as np
>>> from alphaforge.evaluation.metrics import RMSE, MAE, DirectionalAccuracy

Compute a single metric:

>>> rmse = RMSE()
>>> rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
0.5773502691896258

Check correct sign fraction:

>>> da = DirectionalAccuracy()
>>> da(np.array([0.5, -0.3, 1.2]), np.array([0.1, 0.4, 0.8]))
0.6666666666666666

Use the protocol for runtime type checks:

>>> from alphaforge.evaluation.metrics import MetricFn
>>> isinstance(RMSE(), MetricFn)
True

See Also
--------
nowcast_data.models.evaluation.decompose_accuracy_by_horizon :
    Publication-date-anchored horizon decomposition that accepts ``MetricFn``.
nowcast_data.models.evaluation.benchmark_evaluation_suite :
    Full metrics x truths x horizons evaluation.
nowcast_data.utils.metrics.compute_forecast_metrics :
    Low-level metric computation that accepts optional ``MetricFn`` sequence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MetricFn(Protocol):
    """Protocol for forecast accuracy metrics.

    Any object with a ``name`` property and the correct call signature
    satisfies this protocol.  The ``@runtime_checkable`` decorator enables
    ``isinstance(obj, MetricFn)`` checks at runtime, which evaluation
    functions use to validate user-supplied metrics.

    Attributes
    ----------
    name : str
        Short, snake_case identifier used as the column header in result
        DataFrames (e.g. ``"rmse"``, ``"directional_accuracy"``).

    Parameters (when called)
    ------------------------
    y_pred : np.ndarray
        1-D array of predicted values.
    y_true : np.ndarray
        1-D array of ground-truth values, same length as *y_pred*.

    Returns (when called)
    ---------------------
    float
        Scalar metric value.  Return ``np.nan`` when the metric is
        undefined for the given inputs (e.g. MAPE with all-zero truths).

    Examples
    --------
    Implement a custom metric:

    >>> class MedianAbsoluteError:
    ...     name = "median_ae"
    ...     def __call__(self, y_pred, y_true):
    ...         return float(np.median(np.abs(y_pred - y_true)))
    >>> isinstance(MedianAbsoluteError(), MetricFn)
    True
    """

    @property
    def name(self) -> str: ...

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------

class RMSE:
    """Root Mean Squared Error.

    .. math::

        \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i)^2}

    The standard accuracy metric in the nowcasting literature.  Penalizes
    large errors more than :class:`MAE` due to the squaring term.

    Examples
    --------
    >>> RMSE()(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    0.5773502691896258
    """

    name: str = "rmse"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def __repr__(self) -> str:
        return "RMSE()"


class MAE:
    """Mean Absolute Error.

    .. math::

        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |\\hat{y}_i - y_i|

    More robust to outliers than :class:`RMSE`.  GDPNow reports both
    RMSE (1.17) and MAE (0.77) for 2011--2025.

    Examples
    --------
    >>> MAE()(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    0.3333333333333333
    """

    name: str = "mae"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def __repr__(self) -> str:
        return "MAE()"


class DirectionalAccuracy:
    """Fraction of predictions with the correct sign.

    .. math::

        \\text{DA} = \\frac{1}{n} \\sum_{i=1}^{n}
        \\mathbb{1}[\\text{sign}(\\hat{y}_i) = \\text{sign}(y_i)]

    Critical for recession detection — tells you whether the model
    correctly identifies positive vs. negative GDP growth.  Reported by
    the IMF (WP/2025/252) and ECB as a key evaluation criterion.

    Notes
    -----
    When both ``y_pred`` and ``y_true`` are zero, ``np.sign`` returns 0
    for both, so the pair counts as a correct prediction.

    Examples
    --------
    >>> DirectionalAccuracy()(np.array([1, -1, 1]), np.array([1, 1, -1]))
    0.3333333333333333
    """

    name: str = "directional_accuracy"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        correct = np.sign(y_pred) == np.sign(y_true)
        return float(np.mean(correct))

    def __repr__(self) -> str:
        return "DirectionalAccuracy()"


class MAPE:
    """Mean Absolute Percentage Error.

    .. math::

        \\text{MAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
        \\left| \\frac{\\hat{y}_i - y_i}{y_i} \\right|

    Provides scale-independent accuracy, useful for cross-country
    comparisons (e.g. IMF cross-country nowcast evaluations).

    Returns ``nan`` when all true values are near-zero (``|y_i| < 1e-10``),
    since the metric is undefined in that case.

    Notes
    -----
    Observations where ``|y_true| < 1e-10`` are excluded from the
    computation to avoid division by zero.  If *all* observations are
    excluded, the result is ``nan``.

    Examples
    --------
    >>> MAPE()(np.array([1.1, 2.2]), np.array([1.0, 2.0]))
    0.1
    >>> import math
    >>> math.isnan(MAPE()(np.array([1.0]), np.array([0.0])))
    True
    """

    name: str = "mape"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        mask = np.abs(y_true) > 1e-10
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))

    def __repr__(self) -> str:
        return "MAPE()"


class MeanError:
    """Signed mean error (bias).

    .. math::

        \\text{ME} = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i)

    Positive values indicate the forecast is systematically too high;
    negative values indicate systematic under-prediction.  An unbiased
    forecast has ``MeanError ≈ 0``.

    Useful for diagnosing whether a model tends to over-predict or
    under-predict GDP growth.

    Examples
    --------
    >>> MeanError()(np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    1.0
    """

    name: str = "mean_error"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(y_pred - y_true))

    def __repr__(self) -> str:
        return "MeanError()"


# ---------------------------------------------------------------------------
# Pre-built suites
# ---------------------------------------------------------------------------

DEFAULT_METRICS: tuple[MetricFn, ...] = (RMSE(), MAE(), DirectionalAccuracy())
"""Default metric suite: RMSE, MAE, and Directional Accuracy.

Used by evaluation functions when no explicit ``metrics`` argument is
provided.  Covers the two most common point-forecast accuracy measures
plus sign correctness.
"""

BENCHMARK_METRICS: tuple[MetricFn, ...] = (
    RMSE(), MAE(), DirectionalAccuracy(), MeanError(), MAPE(),
)
"""Extended suite for benchmark comparison with published results.

Adds :class:`MeanError` (bias detection) and :class:`MAPE`
(scale-independent accuracy) to the default suite.  Matches the metric
set commonly reported across GDPNow, NY Fed, IMF, and ECB publications.
"""
