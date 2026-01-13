"""AlphaForge: general-purpose data/feature management for financial ML."""

from .data.context import DataContext
from .data.query import Query
from .data.schema import TableSchema
from .data.panel import PanelFrame
from .data.universe import Universe, EntityMetadata

from .time.calendar import TradingCalendar
from .time.grids import Grid, SessionGrid, NativeGrid, EventGrid
from .time.align import AlignSpec, AlignedPanel, AvailabilityState, align_panel

from .features.frame import FeatureFrame, Artifact
from .features.template import ParamSpec, SliceSpec, FeatureTemplate
from .features.realization import FeatureRealization, FitState
from .store.cache import MaterializationPolicy
from .store.duckdb_parquet import DuckDBParquetStore
from .features.ops import materialize, join_feature_frames

__all__ = [
    "DataContext",
    "Query",
    "TableSchema",
    "PanelFrame",
    "Universe",
    "EntityMetadata",
    "TradingCalendar",
    "Grid",
    "SessionGrid",
    "NativeGrid",
    "EventGrid",
    "AlignSpec",
    "AlignedPanel",
    "AvailabilityState",
    "align_panel",
    "FeatureFrame",
    "Artifact",
    "ParamSpec",
    "SliceSpec",
    "FeatureTemplate",
    "FeatureRealization",
    "FitState",
    "MaterializationPolicy",
    "materialize",
    "DuckDBParquetStore",
    "join_feature_frames",
]
