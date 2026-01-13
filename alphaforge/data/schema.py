from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: Sequence[str]
    canonical_columns: Sequence[str]
    entity_column: str = "entity_id"
    time_column: str = "date"

    # native resolution and cadence
    native_freq: Optional[str] = None  # "B","D","W","M","Q"
    time_semantics: Optional[str] = None  # "point","interval_end","interval_avg"
    expected_cadence_days: Optional[int] = None

    # vintage support hooks (DataSource implements this later)
    event_time_column: Optional[str] = None
    release_time_column: Optional[str] = None
    revision_id_column: Optional[str] = None
