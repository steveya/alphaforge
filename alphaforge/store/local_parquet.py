import os, json, pickle
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .store import Store
from ..features.frame import FeatureFrame, Artifact
from ..features.realization import FitState


def _ensure(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _can_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class LocalParquetStore(Store):
    """Local filesystem store.

    - If pyarrow is available: parquet
    - Otherwise: pickle
    """

    root: str

    def __post_init__(self):
        _ensure(self.root)
        for sub in ["frames", "catalogs", "meta", "states"]:
            _ensure(os.path.join(self.root, sub))

    def _paths(self, rid: str) -> dict:
        safe = rid.replace(":", "_")
        return {
            "X_parquet": os.path.join(self.root, "frames", f"{safe}.parquet"),
            "X_pickle": os.path.join(self.root, "frames", f"{safe}.pkl"),
            "cat_parquet": os.path.join(self.root, "catalogs", f"{safe}.parquet"),
            "cat_pickle": os.path.join(self.root, "catalogs", f"{safe}.pkl"),
            "meta_json": os.path.join(self.root, "meta", f"{safe}.json"),
        }

    def exists_frame(self, realization_id: str) -> bool:
        p = self._paths(realization_id)
        return os.path.exists(p["meta_json"]) and (
            os.path.exists(p["X_parquet"]) or os.path.exists(p["X_pickle"])
        )

    def get_frame(self, realization_id: str) -> Optional[FeatureFrame]:
        if not self.exists_frame(realization_id):
            return None
        p = self._paths(realization_id)
        use_parquet = _can_parquet() and os.path.exists(p["X_parquet"])

        if use_parquet:
            X = pd.read_parquet(p["X_parquet"])
            cat = (
                pd.read_parquet(p["cat_parquet"])
                if os.path.exists(p["cat_parquet"])
                else pd.DataFrame()
            )
        else:
            with open(p["X_pickle"], "rb") as f:
                X = pickle.load(f)
            with open(p["cat_pickle"], "rb") as f:
                cat = pickle.load(f)

        with open(p["meta_json"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        if (
            isinstance(cat, pd.DataFrame)
            and "feature_id" in cat.columns
            and cat.index.name != "feature_id"
        ):
            cat = cat.set_index("feature_id")

        return FeatureFrame(X=X, catalog=cat, meta=meta, artifacts=None)

    def put_frame(self, realization_id: str, frame: FeatureFrame) -> None:
        p = self._paths(realization_id)
        use_parquet = _can_parquet()

        if use_parquet:
            frame.X.to_parquet(p["X_parquet"])
            frame.catalog.to_parquet(p["cat_parquet"])
        else:
            with open(p["X_pickle"], "wb") as f:
                pickle.dump(frame.X, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(p["cat_pickle"], "wb") as f:
                pickle.dump(frame.catalog, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(p["meta_json"], "w", encoding="utf-8") as f:
            json.dump(frame.meta, f, default=str, indent=2)

    def put_state(self, state: FitState, payload: bytes) -> Artifact:
        safe = state.state_id.replace(":", "_")
        path = os.path.join(self.root, "states", f"{safe}.bin")
        with open(path, "wb") as f:
            f.write(payload)
        return Artifact(
            artifact_id=state.state_id, kind="fit_state", uri=path, meta=state.meta
        )

    def get_state(self, state_id: str) -> bytes:
        safe = state_id.replace(":", "_")
        path = os.path.join(self.root, "states", f"{safe}.bin")
        with open(path, "rb") as f:
            return f.read()
