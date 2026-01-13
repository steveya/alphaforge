from dataclasses import dataclass
from typing import Optional, Sequence, Literal
import pandas as pd


VintageMode = Literal["latest", "first", "specific"]


def _ts_utc(x: Optional[pd.Timestamp | str]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class Query:
    table: str
    columns: Sequence[str]

    # inputs may be naive or timezone-aware; coerce to tz-aware UTC in __post_init__
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    entities: Optional[Sequence[str]] = None

    asof: Optional[pd.Timestamp] = None
    vintage: VintageMode = "latest"
    vintage_id: Optional[str] = None

    grid: Optional[str] = None

    def __post_init__(self) -> None:
        # coerce timestamps to tz-aware UTC
        object.__setattr__(self, "start", _ts_utc(self.start))
        object.__setattr__(self, "end", _ts_utc(self.end))
        object.__setattr__(self, "asof", _ts_utc(self.asof))
