from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Protocol, Tuple, TYPE_CHECKING
import pandas as pd

from ..data.query import Query

if TYPE_CHECKING:
    from ..data.context import DataContext
    from .frame import FeatureFrame
    from .realization import FitState


@dataclass(frozen=True)
class ParamSpec:
    type: str  # "int","float","categorical","bool"
    default: Any
    low: Any | None = None
    high: Any | None = None
    choices: Optional[Sequence[Any]] = None
    log: bool = False


@dataclass(frozen=True)
class SliceSpec:
    start: pd.Timestamp
    end: pd.Timestamp
    entities: Optional[Sequence[str]] = None
    asof: Optional[pd.Timestamp] = None
    grid: Optional[str] = None


class FeatureTemplate(Protocol):
    name: str
    version: str
    param_space: Dict[str, ParamSpec]

    def requires(self, params: Dict[str, Any]) -> list[Tuple[str, Query]]: ...

    # stateful support (fit on train only)
    def fit(
        self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec
    ) -> Optional[FitState]:
        return None

    def transform(
        self,
        ctx: DataContext,
        params: Dict[str, Any],
        slice: SliceSpec,
        state: Optional[FitState],
    ) -> FeatureFrame: ...
