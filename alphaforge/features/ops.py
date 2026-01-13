from typing import Optional, List
import pickle
import pandas as pd

from ..data.context import DataContext
from ..store.cache import MaterializationPolicy
from .template import FeatureTemplate, SliceSpec
from .realization import FeatureRealization
from .frame import FeatureFrame
from .dag import LineageGraph


def materialize(
    ctx: DataContext,
    template: FeatureTemplate,
    realization: FeatureRealization,
    policy: MaterializationPolicy,
    lineage: Optional[LineageGraph] = None,
    fit_slice: Optional[SliceSpec] = None,
) -> FeatureFrame:
    """Materialize a FeatureRealization with caching and (optional) stateful fit."""
    rid = realization.id()

    if lineage is not None:
        lineage.add(
            rid,
            {
                "kind": "feature",
                "template": template.name,
                "version": template.version,
                "params": realization.params,
                "slice": realization.slice.__dict__,
            },
        )

    got = ctx.store.get_frame(rid)
    if got is not None:
        return got

    # stateful fit (optional)
    state = None
    try:
        st = template.fit(ctx, realization.params, fit_slice or realization.slice)  # type: ignore
    except Exception:
        st = None

    if st is not None:
        art = ctx.store.put_state(
            st, pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL)
        )
        if lineage is not None:
            lineage.add(
                art.artifact_id, {"kind": "artifact", "type": art.kind, "uri": art.uri}
            )
            lineage.link(rid, art.artifact_id)
        state = st

    frame = template.transform(ctx, realization.params, realization.slice, state)  # type: ignore
    frame.meta = {
        **frame.meta,
        "realization_id": rid,
        "template": realization.template,
        "version": realization.version,
        "params": realization.params,
        "slice": realization.slice.__dict__,
        "upstream_snapshot": realization.upstream_snapshot,
    }
    frame.validate()

    if policy.persist_mode == "always":
        ctx.store.put_frame(rid, frame)

    return frame


def join_feature_frames(
    frames: List[FeatureFrame], join: str = "inner"
) -> FeatureFrame:
    if not frames:
        raise ValueError("No frames.")

    idx = frames[0].X.index
    if join == "inner":
        for f in frames[1:]:
            idx = idx.intersection(f.X.index)
    elif join == "outer":
        for f in frames[1:]:
            idx = idx.union(f.X.index)
    else:
        raise ValueError("join must be inner/outer")

    X = pd.concat([f.X.reindex(idx) for f in frames], axis=1)
    catalog = pd.concat([f.catalog for f in frames], axis=0)

    if catalog.index.name != "feature_id" and "feature_id" in catalog.columns:
        catalog = catalog.set_index("feature_id")
    catalog = catalog[~catalog.index.duplicated(keep="first")]

    artifacts = []
    for f in frames:
        if f.artifacts:
            artifacts.extend(f.artifacts)

    return FeatureFrame(
        X=X,
        catalog=catalog,
        meta={"joined": [f.meta.get("realization_id") for f in frames]},
        artifacts=(artifacts or None),
    )
