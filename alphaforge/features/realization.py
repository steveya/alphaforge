from dataclasses import dataclass
from typing import Any, Dict, Optional
import json, hashlib

from .template import SliceSpec


def _stable_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))


@dataclass
class FitState:
    state_id: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class FeatureRealization:
    template: str
    version: str
    params: Dict[str, Any]
    slice: SliceSpec
    upstream_snapshot: Optional[str] = None

    def id(self) -> str:
        payload = {
            "template": self.template,
            "version": self.version,
            "params": self.params,
            "slice": {
                "start": str(self.slice.start),
                "end": str(self.slice.end),
                "entities": (
                    list(self.slice.entities)
                    if self.slice.entities is not None
                    else None
                ),
                "asof": str(self.slice.asof) if self.slice.asof is not None else None,
                "grid": self.slice.grid,
            },
            "upstream_snapshot": self.upstream_snapshot,
        }
        h = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
        return f"{self.template}:{self.version}:{h[:16]}"
