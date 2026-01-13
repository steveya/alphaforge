from dataclasses import dataclass
from typing import Literal


PersistMode = Literal["never", "selected", "always"]


@dataclass(frozen=True)
class MaterializationPolicy:
    cache_ephemeral: bool = True
    persist_mode: PersistMode = "selected"
