# alphaforge/features/dataset_spec.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from .target_template import TargetTemplate


@dataclass(frozen=True)
class UniverseSpec:
    """Which entities we are building the panel for."""

    entities: Sequence[str]


@dataclass(frozen=True)
class TimeSpec:
    """Global time/grid settings for the dataset build."""

    start: pd.Timestamp
    end: pd.Timestamp
    calendar: str = "XNYS"
    grid: str = "B"  # "B" daily business day grid; later can be richer
    asof: Optional[pd.Timestamp] = None  # optional global asof cut (PIT); can be None


@dataclass(frozen=True)
class SliceOverride:
    """
    Optional per-feature overrides.
    - lookback: extend the *data fetch* window backwards to support lagged features.
    - grid: override the grid for this feature family.
    - asof: override global asof (rare; useful for PIT macro)
    """

    lookback: Optional[pd.Timedelta] = None
    grid: Optional[str] = None
    asof: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class FeatureRequest:
    """
    A request to materialize a FeatureTemplate with params, optionally with slice overrides.
    """

    template: Any  # FeatureTemplate
    params: Dict[str, Any] = field(default_factory=dict)
    slice_override: Optional[SliceOverride] = None
    key: Optional[str] = None  # purely organizational (grouping/reporting)
    # New: arbitrary tags to annotate all features produced by this request.
    # These will be stamped into the FeatureFrame.catalog as both dict and JSON.
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRequest:
    """
    Target builder (can be implemented as a FeatureTemplate that returns a 1-col frame or Series).
    """

    template: TargetTemplate  # TargetTemplate-like (we will support FeatureTemplate returning 1-col FeatureFrame)
    params: Dict[str, Any] = field(default_factory=dict)
    horizon: int = 1
    name: str = "target"
    slice_override: Optional[SliceOverride] = None


@dataclass(frozen=True)
class JoinPolicy:
    """
    How to join feature frames across families.
    """

    how: str = "inner"  # "inner" safest; "outer" allowed
    sort_index: bool = True


@dataclass(frozen=True)
class MissingnessPolicy:
    """
    What to do with missing values after join.
    """

    final_row_policy: str = "drop_if_any_nan"  # or "keep"


@dataclass(frozen=True)
class DatasetSpec:
    universe: UniverseSpec
    time: TimeSpec
    target: TargetRequest
    features: Sequence[FeatureRequest] = field(default_factory=list)

    join_policy: JoinPolicy = field(default_factory=JoinPolicy)
    missingness: MissingnessPolicy = field(default_factory=MissingnessPolicy)

    name: str = "dataset"
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetArtifact:
    """
    Returned by build_dataset().
    """

    X: pd.DataFrame
    y: pd.Series
    catalog: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(
        default_factory=dict
    )  # optional extra outputs (returns, etc.)
