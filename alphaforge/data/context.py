from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from .source import DataSource
from .query import Query
from .panel import PanelFrame
from .universe import Universe, EntityMetadata
from ..time.calendar import TradingCalendar
from ..store.store import Store


@dataclass
class DataContext:
    sources: Mapping[str, DataSource]
    calendars: Mapping[str, TradingCalendar]
    store: Store
    universe: Optional[Universe] = None
    entity_meta: Optional[EntityMetadata] = None

    def fetch_panel(self, source: str, q: Query) -> PanelFrame:
        df = self.sources[source].fetch(q)
        panel = PanelFrame.from_long(df)

        # If the source provides a schema and it's a daily/business series,
        # map the input date labels to session close UTC timestamps using calendar.
        try:
            schema = self.sources[source].schemas().get(q.table)
        except Exception:
            schema = None

        if schema is not None and schema.native_freq in ("B", "D"):
            # determine calendar: prefer q.grid like 'sessions:XNYS' else default 'XNYS'
            cal_name = "XNYS"
            if q.grid and isinstance(q.grid, str) and ":" in q.grid:
                parts = q.grid.split(":", 1)
                if len(parts) == 2:
                    cal_name = parts[1]
            if cal_name in self.calendars:
                cal = self.calendars[cal_name]
                dates = pd.DatetimeIndex(panel.df.index.get_level_values("ts_utc"))
                entities = panel.df.index.get_level_values("entity_id")
                closes = [cal.session_close_utc(d) for d in dates]
                new_index = pd.MultiIndex.from_arrays(
                    [pd.DatetimeIndex(closes), entities], names=["ts_utc", "entity_id"]
                )
                panel = PanelFrame(panel.df.copy())
                panel.df.index = new_index

        # first apply start/end/entities pushdowns
        panel = panel.slice(start=q.start, end=q.end, entities=q.entities)
        # then enforce as-of filtration: drop any rows with ts_utc > q.asof
        if q.asof is not None:
            panel = panel.slice(end=q.asof)
        if self.universe is not None:
            panel = self.universe.restrict_panel(panel)  # type: ignore
        return panel.ensure_sorted()
