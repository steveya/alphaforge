"""AlphaForge: general-purpose data/feature management for financial ML."""

from .data.context import DataContext
from .data.panel import PanelFrame
from .data.query import Query
from .data.schema import TableSchema
from .data.universe import EntityMetadata, Universe
from .features.frame import Artifact, FeatureFrame
from .features.ops import join_feature_frames, materialize
from .features.realization import FeatureRealization, FitState
from .features.template import FeatureTemplate, ParamSpec, SliceSpec
from .pit.accessor import PITAccessor
from .pit.ref_entity import make_ref_entity_id, parse_ref_entity_id
from .store.cache import MaterializationPolicy
from .store.duckdb_parquet import DuckDBParquetStore
from .time.align import AlignedPanel, AlignSpec, AvailabilityState, align_panel
from .time.calendar import TradingCalendar
from .time.grids import EventGrid, Grid, NativeGrid, SessionGrid
from .time.ref_period import RefFreq, RefPeriod

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
    "RefFreq",
    "RefPeriod",
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
    "PITAccessor",
    "make_ref_entity_id",
    "parse_ref_entity_id",
]
