# alphaforge/features/dataset_spec.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence

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

    def __post_init__(self) -> None:
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("TimeSpec.start must be <= TimeSpec.end.")


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


def _merge_slice_overrides(
    parent: Optional["SliceOverride"],
    child: Optional["SliceOverride"],
) -> Optional["SliceOverride"]:
    if parent is None:
        return child
    if child is None:
        return parent
    return SliceOverride(
        lookback=child.lookback if child.lookback is not None else parent.lookback,
        grid=child.grid if child.grid is not None else parent.grid,
        asof=child.asof if child.asof is not None else parent.asof,
    )


def _compose_request_key(parent: Optional[str], child: Optional[str]) -> Optional[str]:
    pieces = [piece for piece in (parent, child) if piece]
    if not pieces:
        return None
    return "/".join(pieces)


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
class FeatureRequestGroup:
    """Composable group of feature requests with inherited metadata."""

    requests: Sequence[FeatureRequest | "FeatureRequestGroup"] = field(
        default_factory=tuple
    )
    slice_override: Optional[SliceOverride] = None
    key: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("FeatureRequestGroup.requests cannot be empty.")


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

    def __post_init__(self) -> None:
        if self.how not in {"inner", "outer"}:
            raise ValueError(
                f"JoinPolicy.how must be 'inner' or 'outer', got {self.how!r}."
            )


@dataclass(frozen=True)
class MissingnessPolicy:
    """
    What to do with missing values after join.
    """

    final_row_policy: str = "drop_if_any_nan"  # or "keep"

    def __post_init__(self) -> None:
        if self.final_row_policy not in {"drop_if_any_nan", "keep"}:
            raise ValueError(
                "MissingnessPolicy.final_row_policy must be "
                f"'drop_if_any_nan' or 'keep', got {self.final_row_policy!r}."
            )


def _flatten_feature_requests(
    features: Sequence[Any],
    *,
    inherited_slice_override: Optional[SliceOverride] = None,
    inherited_key: Optional[str] = None,
    inherited_tags: Optional[Dict[str, Any]] = None,
) -> list[FeatureRequest]:
    flat: list[FeatureRequest] = []
    base_tags = dict(inherited_tags or {})

    for item in features:
        if isinstance(item, FeatureRequest):
            merged_tags = dict(base_tags)
            merged_tags.update(item.tags)
            flat.append(
                replace(
                    item,
                    slice_override=_merge_slice_overrides(
                        inherited_slice_override,
                        item.slice_override,
                    ),
                    key=_compose_request_key(inherited_key, item.key),
                    tags=merged_tags,
                )
            )
            continue

        if isinstance(item, FeatureRequestGroup):
            nested_tags = dict(base_tags)
            nested_tags.update(item.tags)
            flat.extend(
                _flatten_feature_requests(
                    item.requests,
                    inherited_slice_override=_merge_slice_overrides(
                        inherited_slice_override,
                        item.slice_override,
                    ),
                    inherited_key=_compose_request_key(inherited_key, item.key),
                    inherited_tags=nested_tags,
                )
            )
            continue

        raise TypeError(
            "DatasetSpec.features must contain FeatureRequest or "
            f"FeatureRequestGroup items, got {type(item)!r}."
        )

    return flat


@dataclass(frozen=True)
class DatasetSpec:
    universe: UniverseSpec
    time: TimeSpec
    target: TargetRequest
    features: Sequence[FeatureRequest | FeatureRequestGroup] = field(
        default_factory=list
    )

    join_policy: JoinPolicy = field(default_factory=JoinPolicy)
    missingness: MissingnessPolicy = field(default_factory=MissingnessPolicy)

    name: str = "dataset"
    tags: Dict[str, Any] = field(default_factory=dict)

    def feature_requests(self) -> list[FeatureRequest]:
        """Return the flattened feature-request list used by the builder."""
        return _flatten_feature_requests(self.features)


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
