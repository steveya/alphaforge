"""Staleness-aware weight adjustment for multi-source signal combination."""
from __future__ import annotations

from typing import Callable


def adjust_weights_for_health(
    weights: dict[str, float],
    health_fn: Callable[[str], float],
    source_map: dict[str, str],
    renormalize: bool = True,
) -> dict[str, float]:
    """Adjust pipeline weights based on source health weight factors.

    Parameters
    ----------
    weights : Nominal weights per pipeline name.
    health_fn : Callable that takes a source name and returns a weight factor
        in [0, 1]. Typically ``ctx.source_weight``.
    source_map : Mapping from pipeline name to source name.
    renormalize : If True, scale surviving weights so total matches original.

    Returns
    -------
    Effective weights after health adjustment (and optional renormalization).
    """
    effective: dict[str, float] = dict(weights)

    for name in list(effective):
        source = source_map.get(name)
        if source is not None:
            factor = health_fn(source)
            effective[name] = effective[name] * factor

    if renormalize:
        original_total = sum(weights.values())
        current_total = sum(effective.values())
        if current_total > 0 and current_total < original_total:
            scale = original_total / current_total
            effective = {k: v * scale for k, v in effective.items()}

    return effective
