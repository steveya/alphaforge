"""Composable pipeline component protocols.

Three stateless building blocks for research mode:

- **Filter**: removes rows from a DataFrame.
- **Transformer**: adds or modifies columns.
- **Signal**: collapses a panel into a cross-sectional score.

One marker for training mode:

- **Parametric**: component exposes named parameters that can be
  optimized. In research mode these are plain floats; in training
  mode they can be replaced with tensor-backed values.

Two concrete orchestrators:

- **SimplePipeline**: linear chain of filters -> transformers -> signal.
- **PipelineVariant**: a named (filter config, pipeline) pair for model selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from alphaforge.logging import get_logger

_log = get_logger(__name__)


@runtime_checkable
class Filter(Protocol):
    """Remove rows from a DataFrame based on criteria."""

    name: str

    def apply(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame: ...


@runtime_checkable
class Transformer(Protocol):
    """Add or modify columns in a DataFrame."""

    name: str

    def transform(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame: ...


@runtime_checkable
class Signal(Protocol):
    """Collapse a panel into a cross-sectional score (entity -> float)."""

    name: str

    def score(self, df: pd.DataFrame, **params: Any) -> pd.Series: ...


@runtime_checkable
class Parametric(Protocol):
    """Marker for components with tunable parameters.

    Transformers and Signals may implement this. Filters do NOT.
    """

    name: str

    def get_params(self) -> dict[str, float]: ...
    def set_params(self, params: dict[str, float]) -> None: ...


@runtime_checkable
class Pipeline(Protocol):
    """Ordered sequence of components with a terminal signal."""

    name: str

    def run(self, df: pd.DataFrame, **params: Any) -> pd.Series: ...

    def get_all_params(self) -> dict[str, dict[str, float]]: ...

    def set_all_params(self, params: dict[str, dict[str, float]]) -> None: ...


@dataclass
class SimplePipeline:
    """Linear chain: filters -> transformers -> signal."""

    name: str
    filters: list[Filter] = field(default_factory=list)
    transformers: list[Transformer] = field(default_factory=list)
    signal: Signal | None = None

    def run(self, df: pd.DataFrame, **params: Any) -> pd.Series:
        t0 = time.monotonic()
        out = df
        for f in self.filters:
            before = len(out)
            out = f.apply(out, **params)
            _log.debug("filter_applied", pipeline=self.name, filter=f.name, before=before, after=len(out))
            if out.empty:
                _log.info("pipeline_empty_after_filter", pipeline=self.name, filter=f.name)
                return pd.Series(dtype="float64")
        for t in self.transformers:
            out = t.transform(out, **params)
        if self.signal is None:
            raise ValueError(f"Pipeline {self.name!r} has no terminal Signal")
        result = self.signal.score(out, **params)
        duration_ms = int((time.monotonic() - t0) * 1000)
        _log.debug("pipeline_run_complete", pipeline=self.name, entities=len(result), duration_ms=duration_ms)
        return result

    def get_all_params(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for component in [*self.filters, *self.transformers]:
            if isinstance(component, Parametric):
                result[component.name] = component.get_params()
        if self.signal is not None and isinstance(self.signal, Parametric):
            result[self.signal.name] = self.signal.get_params()
        return result

    def set_all_params(self, params: dict[str, dict[str, float]]) -> None:
        for component in [*self.filters, *self.transformers]:
            if isinstance(component, Parametric) and component.name in params:
                component.set_params(params[component.name])
        if self.signal is not None and isinstance(self.signal, Parametric):
            if self.signal.name in params:
                self.signal.set_params(params[self.signal.name])


@dataclass(frozen=True)
class PipelineVariant:
    """A named (filter config, pipeline) pair for model selection."""

    name: str
    pipeline: SimplePipeline
    description: str = ""
