# alphaforge/features/target_template.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

import pandas as pd

from .template import ParamSpec, SliceSpec

if TYPE_CHECKING:
    from ..data.context import DataContext


@dataclass
class TargetFrame:
    """
    Canonical output of a TargetTemplate.
    - y: target series indexed like features (typically MultiIndex: [date, entity_id])
    - meta: free-form metadata (definition, scale, horizon, etc.)
    """

    y: pd.Series
    meta: Dict[str, Any]


class TargetTemplate(Protocol):
    """
    TargetTemplate mirrors FeatureTemplate but returns TargetFrame.
    """

    name: str
    version: str
    param_space: Dict[str, ParamSpec]

    def fit(
        self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec
    ) -> Optional[Any]:
        return None

    def transform(
        self,
        ctx: DataContext,
        params: Dict[str, Any],
        slice: SliceSpec,
        state: Optional[Any],
    ) -> TargetFrame: ...
