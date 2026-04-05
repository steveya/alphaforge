"""Target policy — release-selection logic for PIT series.

Provides :class:`TargetPolicy` and helpers for resolving target values
from point-in-time release streams.  The functions are generic and work
with any :class:`~alphaforge.pit.adapters.base.PITAdapter` implementation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

from alphaforge.pit.adapters.base import PITAdapter
from alphaforge.time.ref_period import RefFreq, RefPeriod, coerce_ref_period

if TYPE_CHECKING:
    from alphaforge.pit.catalog import SeriesCatalog


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TargetPolicy:
    """Configuration for selecting a target value from available releases.

    Attributes:
        mode: Selection strategy.

            * ``"latest_available"`` — pick the most recent non-null release.
            * ``"nth_release"``      — pick exactly the *n*-th non-null release.
            * ``"next_release"``     — signal the rank of the next expected release.

        nth: Required when *mode* is ``"nth_release"``.  Must be >= 1.
        max_release_rank: Optional cap on the number of releases considered.
    """

    mode: Literal["latest_available", "nth_release", "next_release"]
    nth: int | None = None
    max_release_rank: int | None = None


# ---------------------------------------------------------------------------
# Quarter calendar helpers
# ---------------------------------------------------------------------------


def quarter_end_date(ref_quarter: str | pd.Period | RefPeriod) -> date:
    """Return the quarter-end date for a reference quarter.

    Args:
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period
            (e.g., ``"2025Q1"``, ``pd.Period("2025Q1", freq="Q")``).

    Returns:
        The calendar quarter-end date.

    Raises:
        ValueError: If the reference quarter does not match ``YYYYQn``.
    """
    try:
        return coerce_ref_period(ref_quarter, freq=RefFreq.Q).end_obs_date().date()
    except ValueError as exc:
        raise ValueError(
            f"Expected quarterly reference in format YYYYQn, got {ref_quarter!r}"
        ) from exc


def quarter_start_date(ref_quarter: str | pd.Period | RefPeriod) -> date:
    """Return the quarter-start date for a reference quarter."""
    try:
        return coerce_ref_period(ref_quarter, freq=RefFreq.Q).start_obs_date().date()
    except ValueError as exc:
        raise ValueError(
            f"Expected quarterly reference in format YYYYQn, got {ref_quarter!r}"
        ) from exc


def quarter_obs_date(
    ref_quarter: str | pd.Period | RefPeriod,
    obs_date_anchor: Literal["end", "start"] = "end",
) -> date:
    """Return the quarter observation date under the configured anchor policy."""
    return (
        coerce_ref_period(ref_quarter, freq=RefFreq.Q)
        .obs_date(anchor=obs_date_anchor)
        .date()
    )


# ---------------------------------------------------------------------------
# Observation anchor resolution
# ---------------------------------------------------------------------------


def resolve_target_obs_date_anchor(
    target_obs_date_anchor: Literal["auto", "end", "start"],
    *,
    target_series_key: str,
    catalog: SeriesCatalog | None,
) -> Literal["end", "start"]:
    """Resolve the effective target observation anchor.

    Resolution policy:
    1. Explicit ``"start"`` or ``"end"`` is honored.
    2. ``"auto"`` uses catalog metadata ``obs_date_anchor`` for the target series.
    3. If metadata is missing/unknown, fallback is ``"end"``.
    """
    if target_obs_date_anchor in {"start", "end"}:
        return target_obs_date_anchor  # type: ignore[return-value]
    if target_obs_date_anchor != "auto":
        raise ValueError(
            f"target_obs_date_anchor must be one of 'auto', 'start', 'end', got "
            f"{target_obs_date_anchor!r}"
        )
    if catalog is not None:
        meta = catalog.get(target_series_key)
        if meta is not None and meta.obs_date_anchor in {"start", "end"}:
            return meta.obs_date_anchor  # type: ignore[return-value]
    return "end"


# ---------------------------------------------------------------------------
# Release listing
# ---------------------------------------------------------------------------


def list_quarterly_target_releases_asof(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    asof_date: date,
    obs_date_anchor: Literal["end", "start"] = "end",
) -> pd.DataFrame:
    """List available quarterly releases up to an as-of date.

    Args:
        adapter: PIT adapter that supports ``list_pit_observations_asof``.
        series_key: PIT series key for the target series.
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period.
        asof_date: Vintage cut-off date (treated as end-of-day UTC).
        obs_date_anchor: Observation date anchor policy.

    Returns:
        DataFrame sorted by ``asof_utc`` (ascending) with columns:
        ``obs_date`` (quarter start/end per anchor), ``asof_utc`` (release timestamp),
        and ``value``.
    """
    obs_date = quarter_obs_date(ref_quarter, obs_date_anchor=obs_date_anchor)
    releases = adapter.list_pit_observations_asof(
        series_key=series_key,
        obs_date=obs_date,
        asof_date=asof_date,
    )
    releases = releases.loc[:, ["obs_date", "asof_utc", "value"]].copy()
    return releases.sort_values("asof_utc", kind="mergesort").reset_index(drop=True)


def list_quarterly_target_releases_asof_multi(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarters: Iterable[str | pd.Period],
    asof_date: date,
    obs_date_anchor: Literal["end", "start"] = "end",
) -> dict[str, pd.DataFrame]:
    """Batch release lookup for multiple quarterly references at the same as-of date."""
    refs = [str(ref) for ref in ref_quarters]
    if not refs:
        return {}

    if hasattr(adapter, "list_pit_observations_asof_multi"):
        requests = pd.DataFrame(
            {
                "request_id": refs,
                "series_key": [series_key] * len(refs),
                "obs_date": [
                    quarter_obs_date(ref, obs_date_anchor=obs_date_anchor) for ref in refs
                ],
                "asof_date": [asof_date] * len(refs),
            }
        )
        try:
            releases = adapter.list_pit_observations_asof_multi(requests)  # type: ignore[attr-defined]
        except NotImplementedError:
            releases = pd.DataFrame()
        out: dict[str, pd.DataFrame] = {}
        if not releases.empty and "request_id" in releases.columns:
            grouped = releases.groupby("request_id", sort=False)
            for request_id, frame in grouped:
                out[str(request_id)] = (
                    frame.loc[:, ["obs_date", "asof_utc", "value"]]
                    .sort_values("asof_utc", kind="mergesort")
                    .reset_index(drop=True)
                )
        empty = pd.DataFrame(
            {
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )
        for ref in refs:
            out.setdefault(ref, empty.copy())
        return out

    return {
        ref: list_quarterly_target_releases_asof(
            adapter,
            series_key=series_key,
            ref_quarter=ref,
            asof_date=asof_date,
            obs_date_anchor=obs_date_anchor,
        )
        for ref in refs
    }


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def resolve_target_from_releases(
    releases: pd.DataFrame,
    policy: TargetPolicy,
) -> tuple[float | None, dict]:
    """Resolve a target value from available releases.

    Args:
        releases: DataFrame containing ``obs_date``, ``asof_utc``, and ``value`` columns.
        policy: Target policy configuration.

    Returns:
        Tuple of (value, metadata). ``value`` is ``None`` if the policy cannot select
        a release. Metadata includes:
        ``policy_mode``, ``selected_release_rank``, ``selected_release_asof_utc``,
        ``target_release_rank``, ``available_release_ranks``, ``n_releases_available``,
        and ``obs_date``. ``selected_*`` fields only describe an available release,
        while ``target_release_rank`` is used to signal a future (next) release.

    Raises:
        ValueError: If the policy configuration is invalid.
    """
    releases_sorted = releases.sort_values("asof_utc", kind="mergesort").reset_index(drop=True)
    raw_k = len(releases_sorted)
    obs_date = releases_sorted["obs_date"].iloc[0] if raw_k > 0 else None

    nonnull_releases_all = releases_sorted.loc[releases_sorted["value"].notna()].copy()
    nonnull_releases_all["__row_rank"] = nonnull_releases_all.index + 1

    if policy.max_release_rank is not None:
        if policy.max_release_rank < 1:
            raise ValueError("max_release_rank must be >= 1")
        nonnull_releases = nonnull_releases_all.head(policy.max_release_rank).reset_index(drop=True)
    else:
        nonnull_releases = nonnull_releases_all.reset_index(drop=True)

    k = len(nonnull_releases)
    available_release_ranks = list(range(1, k + 1))
    nonnull_release_row_ranks = nonnull_releases["__row_rank"].astype(int).tolist()
    n_nonnull = k

    value: float | None = None
    selected_rank: int | None = None
    selected_row_rank: int | None = None
    selected_asof_utc = None
    target_release_rank: int | None = None

    if policy.mode == "latest_available":
        if n_nonnull > 0:
            selected_rank = n_nonnull
            row = nonnull_releases.iloc[selected_rank - 1]
            value = float(row["value"])
            selected_asof_utc = row["asof_utc"]
            selected_row_rank = int(row["__row_rank"])
        elif raw_k > 0:
            # Preserve historical metadata semantics: latest_available reports rank=1 when
            # release rows exist but all values are null.
            selected_rank = 1
    elif policy.mode == "nth_release":
        if policy.nth is None:
            raise ValueError("TargetPolicy.nth must be set for nth_release mode")
        if policy.nth < 1:
            raise ValueError("TargetPolicy.nth must be >= 1")
        selected_rank = policy.nth
        if policy.nth <= n_nonnull:
            row = nonnull_releases.iloc[selected_rank - 1]
            value = float(row["value"])
            selected_asof_utc = row["asof_utc"]
            selected_row_rank = int(row["__row_rank"])
    elif policy.mode == "next_release":
        target_release_rank = n_nonnull + 1
        if policy.max_release_rank is not None:
            target_release_rank = min(target_release_rank, policy.max_release_rank)
    else:
        raise ValueError(f"Unknown target policy mode: {policy.mode}")

    meta = {
        "policy_mode": policy.mode,
        "selected_release_rank": selected_rank,
        "selected_release_asof_utc": selected_asof_utc if value is not None else None,
        "selected_release_row_rank": selected_row_rank if value is not None else None,
        "target_release_rank": target_release_rank,
        "available_release_ranks": available_release_ranks,
        "available_nonnull_release_row_ranks": nonnull_release_row_ranks,
        "n_releases_available": k,
        "n_nonnull_releases_available": n_nonnull,
        "n_raw_releases_available": raw_k,
        "obs_date": obs_date,
    }
    return value, meta


# ---------------------------------------------------------------------------
# High-level resolution
# ---------------------------------------------------------------------------


def resolve_quarterly_final_target(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    evaluation_asof_date: date,
    policy: TargetPolicy = TargetPolicy(mode="nth_release", nth=3, max_release_rank=3),
    obs_date_anchor: Literal["end", "start"] = "end",
) -> tuple[float | None, dict]:
    """Resolve a final target value for a quarterly reference.

    Args:
        adapter: PIT adapter that supports ``list_pit_observations_asof``.
        series_key: PIT series key for the target series.
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period.
        evaluation_asof_date: As-of date used to resolve the final target proxy.
        policy: Target selection policy (defaults to nth_release=3 capped to 3 releases).
        obs_date_anchor: Observation date anchor policy.

    Returns:
        Tuple of (value, metadata) from ``resolve_target_from_releases``.
    """
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key=series_key,
        ref_quarter=ref_quarter,
        asof_date=evaluation_asof_date,
        obs_date_anchor=obs_date_anchor,
    )
    return resolve_target_from_releases(releases, policy)


def get_quarterly_release_observation_stream(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    asof_date: date,
    max_release_rank: int | None = None,
    obs_date_anchor: Literal["end", "start"] = "end",
) -> pd.DataFrame:
    """Return a time-ordered observation stream of quarterly releases."""
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key=series_key,
        ref_quarter=ref_quarter,
        asof_date=asof_date,
        obs_date_anchor=obs_date_anchor,
    )
    if max_release_rank is not None:
        if max_release_rank < 1:
            raise ValueError("max_release_rank must be >= 1")
        releases = releases.head(max_release_rank).reset_index(drop=True)

    if releases.empty:
        return pd.DataFrame(
            {
                "series_key": pd.Series(dtype="object"),
                "ref_quarter": pd.Series(dtype="object"),
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "release_rank": pd.Series(dtype="int64"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )

    release_rank = pd.Series(range(1, len(releases) + 1), name="release_rank")
    stream = releases.copy()
    stream.insert(0, "series_key", series_key)
    stream.insert(1, "ref_quarter", str(ref_quarter))
    stream.insert(3, "release_rank", release_rank)
    return stream.loc[
        :, ["series_key", "ref_quarter", "obs_date", "release_rank", "asof_utc", "value"]
    ]
