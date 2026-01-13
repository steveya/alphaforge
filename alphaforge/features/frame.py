from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pandas as pd
import json  # added


@dataclass
class Artifact:
    artifact_id: str
    kind: str
    uri: str
    meta: Dict[str, Any]


@dataclass
class FeatureFrame:
    """X + catalog + meta (+ optional artifacts)."""

    X: pd.DataFrame
    catalog: pd.DataFrame  # index=feature_id recommended
    meta: Dict[str, Any]
    artifacts: Optional[List[Artifact]] = None

    def validate(self) -> None:
        if not isinstance(self.X.index, pd.MultiIndex):
            raise ValueError("FeatureFrame.X must have MultiIndex (date, entity_id).")
        cols = list(self.X.columns)
        if len(cols) and len(self.catalog):
            if (
                self.catalog.index.name != "feature_id"
                and "feature_id" in self.catalog.columns
            ):
                self.catalog = self.catalog.set_index("feature_id")
            missing = [c for c in cols if c not in self.catalog.index]
            if missing:
                raise ValueError(
                    f"Catalog missing {len(missing)} features (e.g. {missing[:3]})."
                )

    def set_tags(self, tags: Dict[str, Any], overwrite: bool = True) -> "FeatureFrame":
        """
        Broadcast tags to all rows in catalog:
        - catalog['tags'] holds the dict (in-memory convenience)
        - catalog['tags_json'] holds the JSON string (for persistence)
        If overwrite=False, merges with any existing dict; request tags override.
        """
        if self.catalog is None or self.catalog.empty:
            return self
        cat = self.catalog.copy()
        # Ensure columns exist
        if "tags" not in cat.columns:
            cat["tags"] = None
        if "tags_json" not in cat.columns:
            cat["tags_json"] = None

        if overwrite:
            merged_tags = [dict(tags)] * len(cat)
        else:
            merged_tags = []
            for existing in cat["tags"].tolist():
                existing_dict = existing if isinstance(existing, dict) else {}
                upd = dict(existing_dict)
                upd.update(tags or {})
                merged_tags.append(upd)

        cat["tags"] = merged_tags
        cat["tags_json"] = [
            json.dumps(t, sort_keys=True) if isinstance(t, dict) else None
            for t in merged_tags
        ]
        self.catalog = cat
        return self
