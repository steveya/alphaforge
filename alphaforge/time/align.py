from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal
import pandas as pd

from ..data.panel import PanelFrame, TS, ENTITY
from ..data.schema import TableSchema
from .grids import Grid


class AvailabilityState(str, Enum):
    AVAILABLE = "available"
    NO_UPDATE_EXPECTED = (
        "no_update_expected"  # structural (native low-freq, between releases)
    )
    NOT_YET_RELEASED = (
        "not_yet_released"  # requires DataSource vintage logic; placeholder in MVP
    )
    TEMPORARY_OUTAGE = "temporary_outage"  # abnormal missingness
    DISCONTINUED = "discontinued"  # abnormal missingness
    MISSING_UNKNOWN = "missing_unknown"  # abnormal missingness


AlignMethod = Literal["none", "ffill", "step_hold", "interp"]


@dataclass(frozen=True)
class AlignSpec:
    target_grid: str
    method: AlignMethod = "ffill"
    respect_asof: bool = True
    add_missingness_channels: bool = True
    max_fill_gap_days: int | None = None
    cadence_threshold_mult: float = 2.5


@dataclass
class AlignedPanel:
    value: PanelFrame
    observed: PanelFrame  # bool: "new obs released at this date"
    availability: PanelFrame  # AvailabilityState strings per cell


def _tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(None) if idx.tz is not None else idx


def _ts_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Ensure tz-aware UTC representation for comparisons
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _expected_days(schema: TableSchema) -> Optional[int]:
    if schema.expected_cadence_days is not None:
        return schema.expected_cadence_days
    f = (schema.native_freq or "").upper()
    return {"B": 1, "D": 1, "W": 7, "M": 30, "Q": 90}.get(f)


def align_panel(
    panel: PanelFrame,
    schema: TableSchema,
    grid: Grid,
    align: AlignSpec,
    asof: pd.Timestamp | None = None,
) -> AlignedPanel:
    """Align a PanelFrame to a target Grid and produce missingness typing.

    MVP semantics:
    - structural missingness: NO_UPDATE_EXPECTED for low-freq between observation dates
    - abnormal missingness:
        * daily series: missing -> MISSING_UNKNOWN
        * low-freq series: missing on an observation date -> TEMPORARY_OUTAGE
    - NOT_YET_RELEASED: reserved for true vintage-aware sources (to add next)
    """
    df = panel.df.copy()
    gidx = _ts_utc(pd.DatetimeIndex(grid.index))

    d = _ts_utc(pd.DatetimeIndex(df.index.get_level_values(TS))).normalize()
    e = df.index.get_level_values(ENTITY)
    df.index = pd.MultiIndex.from_arrays([d, e], names=[TS, ENTITY])

    entities = df.index.get_level_values(ENTITY).unique()
    fields = list(df.columns)
    exp = _expected_days(schema)

    vals, obs, avs = [], [], []
    for ent in entities:
        sub = df.xs(ent, level=ENTITY).sort_index()
        base = sub.reindex(gidx)

        # observed = dates where raw source had any value (before alignment)
        obs_dates = sub.index[sub.notna().any(axis=1)]
        obs_on_grid = pd.Series(False, index=gidx)
        obs_on_grid.loc[pd.DatetimeIndex(obs_dates).intersection(gidx)] = True

        availability = pd.DataFrame(
            AvailabilityState.AVAILABLE.value, index=gidx, columns=fields
        )

        # structural missingness
        if exp is not None and exp > 1:
            availability.loc[~obs_on_grid.values, :] = (
                AvailabilityState.NO_UPDATE_EXPECTED.value
            )

        all_nan = base.isna().all(axis=1)

        # abnormal missingness
        if exp is None or exp <= 1:
            availability.loc[all_nan.values, :] = (
                AvailabilityState.MISSING_UNKNOWN.value
            )
        else:
            # low-freq: missing specifically on a supposed observation date is abnormal
            if obs_on_grid.any():
                nan_on_obs = all_nan.loc[obs_on_grid[obs_on_grid].index]
                if len(nan_on_obs):
                    availability.loc[nan_on_obs.index, :] = (
                        AvailabilityState.TEMPORARY_OUTAGE.value
                    )

        # alignment method for values
        val = base.copy()
        if align.method == "none":
            pass
        elif align.method in ("ffill", "step_hold"):
            val = (
                val.ffill(limit=align.max_fill_gap_days)
                if align.max_fill_gap_days is not None
                else val.ffill()
            )
        elif align.method == "interp":
            val = val.interpolate(limit_direction="both")
        else:
            raise ValueError(f"Unknown align.method={align.method}")

        obs_df = pd.DataFrame(False, index=gidx, columns=fields)
        obs_df.loc[obs_on_grid[obs_on_grid].index, :] = True

        # add entity level back
        val.index = pd.MultiIndex.from_product([val.index, [ent]], names=[TS, ENTITY])
        obs_df.index = pd.MultiIndex.from_product(
            [obs_df.index, [ent]], names=[TS, ENTITY]
        )
        availability.index = pd.MultiIndex.from_product(
            [availability.index, [ent]], names=[TS, ENTITY]
        )

        vals.append(val)
        obs.append(obs_df)
        avs.append(availability)

    return AlignedPanel(
        value=PanelFrame(pd.concat(vals).sort_index()),
        observed=PanelFrame(pd.concat(obs).sort_index()),
        availability=PanelFrame(pd.concat(avs).sort_index()),
    )
