# Point-in-Time (PIT) data

Alphaforge supports revised macro series using a canonical Point-in-Time (PIT) table and two query views:

1. **Snapshot view**: a normal time series indexed by `obs_date`, as-of a cutoff time.
2. **Revision timeline**: a series indexed by `asof_utc` that shows revisions for a single `obs_date`.

## Canonical PIT schema

The PIT table is created automatically when you use a DuckDB-backed store (`DuckDBParquetStore`). It has the following schema:

| Column | Type | Notes |
| --- | --- | --- |
| series_key | TEXT | Series identifier (e.g., "GDP") |
| obs_date | TIMESTAMP | Reference period end date |
| asof_utc | TIMESTAMP | Vintage / knowledge time |
| value | DOUBLE | Observed value |
| release_time_utc | TIMESTAMP | Optional release timestamp |
| revision_id | TEXT | Optional revision label |
| source | TEXT | Optional source |
| meta_json | TEXT | Optional JSON-encoded provenance |
| ingested_utc | TIMESTAMP | Default `now()` at insert |

Uniqueness is enforced on `(series_key, obs_date, asof_utc)`.

## Timezone handling

Input timestamps are normalized to timezone-aware UTC on ingestion. Snapshot and revision queries return UTC indexes.

## Usage

```python
import pandas as pd
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.pit.accessor import PITAccessor

store = DuckDBParquetStore(root="./store")
pit = PITAccessor(store.conn())

df = pd.DataFrame(
    {
        "series_key": ["GDP", "GDP", "GDP", "GDP"],
        "obs_date": [
            pd.Timestamp("2024-09-30"),
            pd.Timestamp("2024-09-30"),
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2024-12-31"),
        ],
        "asof_utc": [
            pd.Timestamp("2024-11-15", tz="UTC"),
            pd.Timestamp("2024-12-20", tz="UTC"),
            pd.Timestamp("2025-02-15", tz="UTC"),
            pd.Timestamp("2025-03-28", tz="UTC"),
        ],
        "value": [100.0, 101.0, 200.0, 202.0],
        "source": ["alfred"] * 4,
    }
)

pit.upsert_pit_observations(df)

snapshot = pit.get_snapshot("GDP", pd.Timestamp("2025-01-01", tz="UTC"))
timeline = pit.get_revision_timeline("GDP", pd.Timestamp("2024-12-31", tz="UTC"))
```

## Reference period keys

PIT queries accept reference period keys that map to `obs_date` end timestamps (UTC midnight).

Supported formats:
- Annual: `YYYY` (e.g., `2025`)
- Quarterly: `YYYYQq` (e.g., `2024Q4`)
- Monthly: `YYYY-MM` or `YYYY/MM` (e.g., `2025-01`)
- Month-end date: `YYYY-MM-DD` (interpreted as monthly, e.g., `2025-01-31`)

Canonical formatting:
- Annual: `YYYY`
- Quarterly: `YYYYQq`
- Monthly: `YYYY-MM`

Example usage:

```python
timeline = pit.get_revision_timeline_ref("GDP", "2024Q4")
snapshot = pit.get_snapshot_ref(
    "GDP",
    asof=pd.Timestamp("2025-06-01", tz="UTC"),
    start_ref="2019Q1",
    end_ref="2024Q4",
)
```

Reference period keys map to the end of the observation period:
- `2024Q4` → `2024-12-31 00:00:00+00:00`
- `2025-01` → `2025-01-31 00:00:00+00:00`
- `2025` → `2025-12-31 00:00:00+00:00`
