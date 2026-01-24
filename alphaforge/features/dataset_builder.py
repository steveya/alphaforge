# alphaforge/features/dataset_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import logging
import pandas as pd
import warnings

from .dataset_spec import (
    DatasetArtifact,
    DatasetSpec,
    FeatureRequest,
    SliceOverride,
    TargetRequest,
)
from .target_template import TargetFrame, TargetTemplate
from .template import SliceSpec
from .frame import FeatureFrame
from ..time.events import EventSource  # NEW
from ..time.grids import EventGrid as EventGridType  # reuse existing class

logger = logging.getLogger(__name__)


def _ts(x: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(x)


def _apply_override(
    base: SliceSpec,
    override: Optional[SliceOverride],
) -> SliceSpec:
    if override is None:
        return base

    start = base.start
    if override.lookback is not None:
        start = _ts(start) - override.lookback

    return SliceSpec(
        start=_ts(start),
        end=_ts(base.end),
        entities=base.entities,
        asof=override.asof if override.asof is not None else base.asof,
        grid=override.grid if override.grid is not None else base.grid,
    )


def _maybe_sort(df: pd.DataFrame, sort_index: bool) -> pd.DataFrame:
    return df.sort_index() if sort_index else df


def _join_feature_frames(
    frames: Sequence[FeatureFrame],
    *,
    how: str,
    sort_index: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Joins FeatureFrame.X column-wise and concatenates catalogs row-wise.

    If FeatureFrame provides a join/concat helper, use it. Otherwise fallback to pandas concat.
    """
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    # If your FeatureFrame defines a join method, prefer it.
    # We attempt duck-typing: FeatureFrame.join(other, how=...)
    try:
        acc = frames[0]
        for ff in frames[1:]:
            acc = acc.join(ff, how=how)  # type: ignore[attr-defined]
        X = acc.X
        catalog = acc.catalog
        X = _maybe_sort(X, sort_index)
        catalog = _maybe_sort(catalog, sort_index)
        return X, catalog
    except Exception:
        pass

    # Fallback: align on index with concat(join=...)
    X = pd.concat([ff.X for ff in frames], axis=1, join=how)
    catalog = pd.concat([ff.catalog for ff in frames], axis=0)

    X = _maybe_sort(X, sort_index)
    catalog = _maybe_sort(catalog, sort_index)
    return X, catalog


def _materialize_template(
    ctx,
    template: Any,
    params: Dict[str, Any],
    slice: SliceSpec,
) -> FeatureFrame:
    """
    Materialize one FeatureTemplate.
    Uses template.fit(...) if present, then template.transform(...).
    """
    state = None
    # stateful features (fit on same slice for now; later add fit_slice support)
    fit_fn = getattr(template, "fit", None)
    if callable(fit_fn):
        state = fit_fn(ctx, params, slice)

    return template.transform(ctx, params, slice, state)


def _materialize_target(
    ctx,
    target: TargetRequest,
    slice: SliceSpec,
) -> tuple[pd.Series, dict]:
    """
    Materialize target using TargetTemplate contract.

    Returns
    -------
    y : pd.Series
    meta : dict
        target meta merged with request meta (horizon/name)
    """
    template = target.template
    params = target.params

    state = None
    fit_fn = getattr(template, "fit", None)
    if callable(fit_fn):
        state = fit_fn(ctx, params, slice)

    out = template.transform(ctx, params, slice, state)

    # Strict path: TargetFrame
    if isinstance(out, TargetFrame):
        y = out.y.rename(target.name)
        meta = dict(out.meta)
        meta.update({"horizon": target.horizon, "name": target.name})
        return y, meta

    # Friendly fallback (optional): allow old target-as-feature 1-col frame
    if isinstance(out, FeatureFrame):
        if out.X.shape[1] != 1:
            raise ValueError(
                f"Target FeatureFrame must have 1 column, got {out.X.shape[1]}"
            )
        y = out.X.iloc[:, 0].rename(target.name)
        meta = {
            "horizon": target.horizon,
            "name": target.name,
            "compat": "FeatureFrame",
        }
        return y, meta

    # Friendly fallback: Series
    if isinstance(out, pd.Series):
        y = out.rename(target.name)
        meta = {"horizon": target.horizon, "name": target.name, "compat": "Series"}
        return y, meta

    raise TypeError(f"Unexpected target output type: {type(out)}")


def build_dataset(
    ctx,
    spec: DatasetSpec,
    *,
    persist: bool = True,
) -> DatasetArtifact:
    """
    Build a dataset from feature + target requests.
    """
    entities = list(spec.universe.entities)
    if not entities:
        raise ValueError(
            "DatasetSpec.universe.entities is empty. Provide at least one entity_id."
        )

    base_slice = SliceSpec(
        start=_ts(spec.time.start),
        end=_ts(spec.time.end),
        entities=entities,
        asof=spec.time.asof,
        grid=spec.time.grid,
    )

    # Build evaluation grid (tz-aware UTC) and full MultiIndex (ts_utc x entities)
    from ..time.grids import build_grid_utc

    cal = ctx.calendars.get(spec.time.calendar)
    if cal is None:
        raise KeyError(f"Calendar {spec.time.calendar} not found in context")

    grid_spec = spec.time.grid
    # Resolve grid_idx for str | EventSource | EventGrid
    if isinstance(grid_spec, EventSource):
        grid_idx = grid_spec.events(
            ctx,
            base_slice.start,
            base_slice.end,
            base_slice.entities,
        )
        grid_idx = pd.DatetimeIndex(grid_idx).tz_convert("UTC").sort_values().unique()
    elif isinstance(grid_spec, EventGridType):
        grid_idx = (
            pd.DatetimeIndex(grid_spec.index).tz_convert("UTC").sort_values().unique()
        )
    else:
        # string (existing behavior)
        grid_idx = build_grid_utc(cal, spec.time.start, spec.time.end, grid_spec)

    if len(grid_idx) == 0:
        raise ValueError(
            "No timestamps found for the requested time range/grid. "
            f"calendar={spec.time.calendar}, grid={spec.time.grid}, "
            f"start={spec.time.start}, end={spec.time.end}, entities={entities}"
        )

    full_index = pd.MultiIndex.from_product(
        [grid_idx, entities], names=["ts_utc", "entity_id"]
    )

    if full_index.empty:
        raise ValueError(
            "Empty dataset index after combining timestamps and entities. "
            f"calendar={spec.time.calendar}, grid={spec.time.grid}, "
            f"start={spec.time.start}, end={spec.time.end}, entities={entities}"
        )

    # 1) materialize features
    feature_frames: List[FeatureFrame] = []
    for req in spec.features:
        s = _apply_override(base_slice, req.slice_override)
        ff = _materialize_template(ctx, req.template, req.params, s)

        # Detect template leakage: warn if template returned timestamps beyond the slice.asof
        try:
            tname = getattr(req.template, "name", repr(req.template))
            if s.asof is not None and ff.X is not None and not ff.X.empty:
                times_check = pd.DatetimeIndex(ff.X.index.get_level_values("ts_utc"))
                # If any timestamps are > asof, warn
                if (times_check > s.asof).any():
                    num_leak = int((times_check > s.asof).sum())
                    warnings.warn(
                        f"Template '{tname}' returned {num_leak} rows with timestamps after asof={s.asof}",
                        UserWarning,
                    )
                    # record flag in meta for diagnostics
                    ff.meta = dict(ff.meta)
                    ff.meta["leak_detected"] = True
        except Exception:
            # best-effort only; don't fail dataset creation for detection issues
            pass

        # If feature frame uses session labels (midnight), convert to session-close timestamps
        if ff.X is not None and not ff.X.empty:
            try:
                times = pd.DatetimeIndex(ff.X.index.get_level_values("ts_utc"))
                # heuristic: if times are midnight session labels, convert to closes
                if all(t.hour == 0 for t in times):
                    entities = ff.X.index.get_level_values("entity_id")
                    closes = [cal.session_close_utc(t) for t in times]
                    new_index = pd.MultiIndex.from_arrays(
                        [pd.DatetimeIndex(closes), entities],
                        names=["ts_utc", "entity_id"],
                    )
                    ff = type(ff)(
                        pd.DataFrame(
                            ff.X.values, index=new_index, columns=ff.X.columns
                        ),
                        ff.catalog,
                        ff.meta,
                    )
            except Exception:
                # if index doesn't have expected levels, ignore
                pass
            ff.X = ff.X.reindex(full_index)
        else:
            ff.X = pd.DataFrame(index=full_index)
        feature_frames.append(ff)

    X, catalog = _join_feature_frames(
        feature_frames,
        how=spec.join_policy.how,
        sort_index=spec.join_policy.sort_index,
    )

    # Ensure X covers the full evaluation index
    if len(X) > 0:
        X = X.reindex(full_index)
    else:
        X = pd.DataFrame(index=full_index)

    # 2) materialize target on *base slice* (but allow override)
    target_slice = _apply_override(base_slice, spec.target.slice_override)
    y, target_meta = _materialize_target(ctx, spec.target, target_slice)

    # Ensure y is a Series with MultiIndex [ts_utc, entity_id]
    if isinstance(y, pd.Series):
        if y.index.nlevels == 1:
            times = pd.DatetimeIndex(y.index)
            closes = [cal.session_close_utc(t) for t in times]
            expanded = pd.MultiIndex.from_product(
                [pd.DatetimeIndex(closes), list(spec.universe.entities)],
                names=["ts_utc", "entity_id"],
            )
            y = pd.Series(
                y.values.repeat(len(spec.universe.entities)),
                index=expanded,
                name=y.name,
            )
        # Reindex y to the full evaluation index, then optionally emit debug logs
        y = y.reindex(full_index)
        logger.debug(
            f"build_dataset: y_index_before_reindex name_levels={y.index.names} len={len(y)} sample_index={list(y.index[:3])}"
        )
        logger.debug(f"build_dataset: full_index sample={list(full_index[:3])}")
        logger.debug(
            f"build_dataset: y_after_reindex_nonmiss={y.notna().sum()} len={len(y)}"
        )
    else:
        raise TypeError("Target must be a pd.Series")

    # 4) missingness policy
    if spec.missingness.final_row_policy == "drop_if_any_nan":
        if len(X) > 0:
            mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[mask]
            y = y.loc[mask]
    elif spec.missingness.final_row_policy == "keep":
        pass
    else:
        raise ValueError(
            f"Unknown final_row_policy: {spec.missingness.final_row_policy}"
        )

    meta = {
        "name": spec.name,
        "tags": dict(spec.tags),
        "calendar": spec.time.calendar,
        "grid": spec.time.grid,
        "start": str(spec.time.start),
        "end": str(spec.time.end),
        "n_features": int(X.shape[1]) if len(X) else 0,
        "n_rows": int(len(X)),
        "target": target_meta,
    }

    return DatasetArtifact(X=X, y=y, catalog=catalog, meta=meta)
