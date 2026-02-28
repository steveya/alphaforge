"""AlphaForge: general-purpose data/feature management for financial ML."""

from .data.context import DataContext
from .data.panel import PanelFrame
from .data.pit_source import PITDataSource
from .data.public_web import (
    ANPFuelPricesDataSource,
    B3HistoricalQuotesDataSource,
    BCBSGSDataSource,
    BEADataSource,
    BLSDataSource,
    CFTCWeeklySwapsSource,
    CMEProductSlateSource,
    DestatisGenesisDataSource,
    DTCCPPDSource,
    ECBSDMXDataSource,
    ECWeeklyOilBulletinDataSource,
    EIADataSource,
    EurexRefdataContractsSource,
    EurexStatsDailySource,
    EurostatDataSource,
    EzoicAdRevenueDailySource,
    IBGESidraDataSource,
    LCHCDSClearDailySource,
)
from .data.query import Query
from .data.schema import TableSchema
from .data.universe import EntityMetadata, Universe
from .features.frame import Artifact, FeatureFrame
from .features.ops import join_feature_frames, materialize
from .features.realization import FeatureRealization, FitState
from .features.template import FeatureTemplate, ParamSpec, SliceSpec
from .pit.accessor import PITAccessor
from .pit.exceptions import (
    PITCausalityError,
    PITContractError,
    PITEngineError,
    PITError,
    PITExperimentalFeatureError,
    PITUnsupportedOperationError,
    PITValidationError,
)
from .pit.guards import ReleaseLagPolicy, effective_asof, pit_leakage_report
from .pit.ref_entity import make_ref_entity_id, parse_ref_entity_id
from .pit.tasks import (
    first_vintage_snapshot,
    forward_fill_with_staleness,
    latest_vintage_snapshot,
    qoq,
    revision_deltas,
    revision_events,
    revision_stability,
    snapshot_at_horizon,
    yoy,
)
from .pit.transforms import PITTransformResult, PITTransformSpec
from .pit.validation import PITValidationReport, validate_pit_observations
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
    "PITDataSource",
    "BLSDataSource",
    "BEADataSource",
    "EIADataSource",
    "EurostatDataSource",
    "ECBSDMXDataSource",
    "DestatisGenesisDataSource",
    "ECWeeklyOilBulletinDataSource",
    "IBGESidraDataSource",
    "BCBSGSDataSource",
    "ANPFuelPricesDataSource",
    "B3HistoricalQuotesDataSource",
    "DTCCPPDSource",
    "CMEProductSlateSource",
    "CFTCWeeklySwapsSource",
    "EurexStatsDailySource",
    "EurexRefdataContractsSource",
    "LCHCDSClearDailySource",
    "EzoicAdRevenueDailySource",
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
    "PITError",
    "PITContractError",
    "PITValidationError",
    "PITUnsupportedOperationError",
    "PITExperimentalFeatureError",
    "PITCausalityError",
    "PITEngineError",
    "PITTransformSpec",
    "PITTransformResult",
    "PITValidationReport",
    "validate_pit_observations",
    "ReleaseLagPolicy",
    "effective_asof",
    "pit_leakage_report",
    "first_vintage_snapshot",
    "latest_vintage_snapshot",
    "snapshot_at_horizon",
    "revision_deltas",
    "revision_events",
    "revision_stability",
    "forward_fill_with_staleness",
    "yoy",
    "qoq",
    "make_ref_entity_id",
    "parse_ref_entity_id",
]
