"""Legacy DataSource protocol for compatibility and raw-loader workflows.

New external loading code should prefer ``SourceAdapter`` plus
``DataContext.fetch(...)`` / ``fetch_many(...)`` / ``prefetch(...)``.

``DataSource`` remains useful where Alphaforge still exposes raw long-frame
loaders directly or where existing panel-oriented integrations have not been
migrated yet.
"""

from typing import Protocol

import pandas as pd

from .query import Query
from .schema import TableSchema


class DataSource(Protocol):
    """Compatibility/raw-loader protocol, not the canonical fetch contract."""

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
