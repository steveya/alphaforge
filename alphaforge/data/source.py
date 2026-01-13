from typing import Protocol
import pandas as pd

from .query import Query
from .schema import TableSchema


class DataSource(Protocol):
    name: str

    def schemas(self) -> dict[str, TableSchema]: ...

    def fetch(self, q: Query) -> pd.DataFrame:
        """Return long DataFrame with time/entity columns + requested columns.

        Must apply pushdowns when possible:
        - columns
        - time range
        - entities
        - asof/vintage (if supported)
        """
        ...
