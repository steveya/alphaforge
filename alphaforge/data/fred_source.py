from typing import Dict
import pandas as pd
from alphaforge.data.source import DataSource
from alphaforge.data.schema import TableSchema
from alphaforge.data.query import Query

class FREDDataSource(DataSource):
    """
    Data source for fetching data from FRED.
    """
    name: str = "fred"

    def __init__(self, api_key: str):
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("Please install fredapi to use the FREDDataSource")
        self._fred = Fred(api_key=api_key)

    def schemas(self) -> Dict[str, TableSchema]:
        """
        Returns a dictionary of table schemas.
        For FRED, we can think of each series as a table.
        However, for simplicity, we'll define a single generic schema for all series.
        """
        return {
            "fred_series": TableSchema(
                name="fred_series",
                required_columns=["value"],
                canonical_columns=["value"],
                entity_column="series_id",
                time_column="date",
                native_freq="D",  # Default to daily, can be overridden
                time_semantics="point",
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        """
        Fetch data from FRED based on the given query.
        """
        if q.table != "fred_series":
            raise ValueError(f"Unknown table: {q.table}")

        if not q.entities:
            raise ValueError("Please specify at least one series_id in the entities field.")

        all_series = []
        for series_id in q.entities:
            realtime_start = q.asof.strftime('%Y-%m-%d') if q.asof else None
            series = self._fred.get_series(
                series_id=series_id,
                observation_start=q.start,
                observation_end=q.end,
                realtime_start=realtime_start,
                realtime_end=realtime_start,
            )
            series = series.to_frame(name="value")
            series["series_id"] = series_id
            series.index.name = "date"
            all_series.append(series.reset_index())

        return pd.concat(all_series, ignore_index=True)
