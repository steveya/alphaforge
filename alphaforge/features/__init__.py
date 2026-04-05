# alphaforge/features/__init__.py
from .calendar_flags import CalendarFlagsTemplate
from .event_dates import EventDateTemplate
from .frame import FeatureFrame
from .market import LagReturnsTemplate, RollingVolatilityTemplate
from .template import FeatureTemplate, ParamSpec, SliceSpec

__all__ = [
    "CalendarFlagsTemplate",
    "EventDateTemplate",
    "FeatureFrame",
    "FeatureTemplate",
    "LagReturnsTemplate",
    "ParamSpec",
    "RollingVolatilityTemplate",
    "SliceSpec",
]
