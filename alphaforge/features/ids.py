from typing import Any, Dict, Optional
import json, hashlib


def _sj(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))


def make_feature_id(
    source_table: str,
    source_key: str,
    family: str,
    transform: str,
    params: Dict[str, Any],
    coord: Optional[str] = None,
) -> str:
    payload = {
        "source_table": source_table,
        "source_key": source_key,
        "family": family,
        "transform": transform,
        "params": params,
        "coord": coord,
    }
    h = hashlib.sha256(_sj(payload).encode("utf-8")).hexdigest()[:12]
    base = f"{family}.{transform}.{h}"
    return f"{base}:{coord}" if coord else base


def group_path(family: str, transform: str, params: Dict[str, Any]) -> str:
    # convention: group by key hyperparams
    keys = []
    for k in sorted(params.keys()):
        if k in ("window", "depth", "lags", "k", "price_col"):
            keys.append(f"{k}={params[k]}")
    return f"{family}/{transform}" + ("" if not keys else "/" + ",".join(keys))
