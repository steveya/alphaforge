# First Rate Futures Guide

`alphaforge.futures` ingests local First Rate Data futures contract files into
canonical parquet artifacts and exposes them through a `SourceAdapter`.

## Required configuration

You must provide:

- `source_dir`: directory containing the raw First Rate contract files
- `artifact_root`: directory where Alphaforge writes parquet artifacts and manifests

Optional:

- `metadata_path`: YAML override for root metadata
- `ALPHAFORGE_FRD_FUTURES_CONFIG`: YAML config file path

Environment variables:

- `ALPHAFORGE_FRD_FUTURES_SOURCE_DIR`
- `ALPHAFORGE_FRD_FUTURES_ARTIFACT_ROOT`
- `ALPHAFORGE_FRD_FUTURES_METADATA_PATH`
- `ALPHAFORGE_FRD_FUTURES_CONFIG`

Resolution order is explicit arguments, then YAML config, then environment variables.

## Expected raw structure

V1 expects a flat directory of contract files:

```text
source_dir/
  ES_M24_5min.txt
  ES_U24_5min.txt
  CL_N23_5min.txt
```

Each file must be a headerless six-column CSV-like text file:

```text
timestamp,open,high,low,close,volume
```

Filenames must follow:

```text
<ROOT>_<MONTH><YY>_5min.txt
```

Examples:

- `ES_M24_5min.txt`
- `CL_N23_5min.txt`
- `VX_F25_5min.txt`

## Artifact structure

The loader writes a separate artifact root:

```text
artifact_root/
  manifests/
    entity_manifest.parquet
  contract_5m_sparse/
    data.parquet
  contract_5m_dense/
    data.parquet
  contract_eod/
    data.parquet
  continuous_5m_execution/
    data.parquet
  continuous_eod_research/
    data.parquet
  roll_schedules/
    data.parquet
```

## Dataset definitions

- `futures.contract_5m_sparse`: observed contract bars only
- `futures.contract_5m_dense`: full 5-minute contract grid with synthetic zero-volume gap bars
- `futures.contract_eod`: daily contract OHLCV aggregated by session date
- `futures.continuous_5m_execution`: unadjusted active-contract chain keyed by execution-safe timestamps
- `futures.continuous_eod_research`: ratio-adjusted daily continuous chain for backward-looking research

Key timestamps:

- `bar_start_utc`: source timestamp interpreted in US Eastern and converted to UTC
- `available_at_utc`: `bar_start_utc + 5 minutes`
- `session_date`: the venue-local session close date

## Example config

```yaml
source_dir: /path/to/fut_contract_price_5m
artifact_root: /path/to/alphaforge_futures_store
metadata_path: /path/to/custom_us_root_metadata.yaml
roll_min_consecutive_sessions: 2
```

## Example usage

```python
from alphaforge import FirstRateFuturesLoader, build_first_rate_futures_context

loader = FirstRateFuturesLoader.from_env()
loader.ingest()

ctx = build_first_rate_futures_context(loader.config)
adapter = ctx.adapters["first_rate_futures"]
entities = adapter.list_entities("futures.continuous_eod_research")
print(entities[:5])
```

## Metadata overrides

The package ships with default US futures metadata and a generic US fallback
pattern. If you need tighter venue rules or root-specific overrides, provide a
YAML file through `metadata_path` or `ALPHAFORGE_FRD_FUTURES_METADATA_PATH`.
