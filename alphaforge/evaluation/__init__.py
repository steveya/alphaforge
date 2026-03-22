"""Evaluation primitives — metric protocol and standard implementations.

This package provides the foundational building blocks for forecast
evaluation.  It is intentionally generic (not specific to nowcasting or
PIT data) so that downstream libraries like ``nowcast-data`` can compose
these primitives into domain-specific evaluation pipelines.

Key components:

- :class:`MetricFn` — a ``Protocol`` that any accuracy metric must satisfy.
- Five built-in metric classes: :class:`RMSE`, :class:`MAE`,
  :class:`DirectionalAccuracy`, :class:`MAPE`, :class:`MeanError`.
- Two pre-built suites: :data:`DEFAULT_METRICS` (RMSE + MAE + DA) and
  :data:`BENCHMARK_METRICS` (adds MeanError + MAPE).

Downstream usage (nowcast-data)::

    from alphaforge.evaluation.metrics import BENCHMARK_METRICS
    from nowcast_data.models.evaluation import benchmark_evaluation_suite

    results = benchmark_evaluation_suite(
        predictions,
        truth_definitions={"advance": ("y_true_release_1", 1)},
        metrics=list(BENCHMARK_METRICS),
    )
"""

from .metrics import (
    BENCHMARK_METRICS,
    DEFAULT_METRICS,
    MAPE,
    MAE,
    MeanError,
    MetricFn,
    RMSE,
    DirectionalAccuracy,
)

__all__ = [
    "MetricFn",
    "RMSE",
    "MAE",
    "DirectionalAccuracy",
    "MAPE",
    "MeanError",
    "DEFAULT_METRICS",
    "BENCHMARK_METRICS",
]
