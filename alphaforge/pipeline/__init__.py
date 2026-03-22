"""Composable pipeline component protocols and orchestrators."""

from .health import (
    CFTC_COT_HEALTH_POLICY,
    DTCC_PPD_HEALTH_POLICY,
    HealthPolicy,
    HealthStatus,
    SourceHealthPolicy,
    SourceHealthStatus,
    assess_health,
    assess_source_health,
)
from .protocols import (
    Filter,
    Parametric,
    Pipeline,
    PipelineVariant,
    Signal,
    SimplePipeline,
    Transformer,
)
from .tracker import HealthTracker, SourceHealthTracker
from .weights import adjust_weights_for_health

__all__ = [
    "Filter",
    "Parametric",
    "Pipeline",
    "PipelineVariant",
    "Signal",
    "SimplePipeline",
    "Transformer",
    "SourceHealthPolicy",
    "SourceHealthStatus",
    "assess_source_health",
    "CFTC_COT_HEALTH_POLICY",
    "DTCC_PPD_HEALTH_POLICY",
    "SourceHealthTracker",
    "HealthPolicy",
    "HealthStatus",
    "HealthTracker",
    "assess_health",
    "adjust_weights_for_health",
]
