from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from typing import Any

import pandas as pd

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.queries import RefSnapshotQuery
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _quarter_end_dates(periods: int) -> list[pd.Timestamp]:
    return [
        period.to_timestamp(how="end").normalize()
        for period in pd.period_range("2010Q1", periods=periods, freq="Q")
    ]


def _sample_observations(
    *,
    periods: int,
    series_count: int,
    revisions_per_period: int,
) -> tuple[pd.DataFrame, list[str]]:
    obs_dates = _quarter_end_dates(periods)
    series_keys = [f"SERIES_{index + 1}" for index in range(series_count)]

    rows: list[dict[str, Any]] = []
    for series_index, series_key in enumerate(series_keys):
        for period_index, obs_date in enumerate(obs_dates):
            base_value = float((series_index + 1) * 100 + period_index)
            for revision_index in range(revisions_per_period):
                rows.append(
                    {
                        "series_key": series_key,
                        "obs_date": obs_date,
                        "asof_utc": pd.Timestamp(obs_date, tz="UTC")
                        + pd.Timedelta(days=15 + revision_index * 15),
                        "value": base_value + revision_index * 0.1,
                    }
                )
    return pd.DataFrame(rows), series_keys


def _time_call(fn, *, iterations: int) -> list[float]:
    fn()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def run_pit_contract_benchmarks(
    *,
    iterations: int = 5,
    periods: int = 40,
    series_count: int = 3,
    revisions_per_period: int = 2,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="alphaforge-bench-") as tmpdir:
        store = DuckDBParquetStore(root=tmpdir)
        pit = PITAccessor(store.conn())
        observations, series_keys = _sample_observations(
            periods=periods,
            series_count=series_count,
            revisions_per_period=revisions_per_period,
        )
        pit.upsert_pit_observations(observations)

        asof = observations["asof_utc"].max()
        start_ref = pd.Period(observations["obs_date"].min(), freq="Q")
        end_ref = pd.Period(observations["obs_date"].max(), freq="Q")
        snapshot_query = RefSnapshotQuery(
            series_key=series_keys[0],
            asof=asof,
            start_ref=start_ref,
            end_ref=end_ref,
        )
        panel_specs = [
            {
                "series_key": series_key,
                "alias": series_key.lower(),
                "start_ref": start_ref,
                "end_ref": end_ref,
                "freq": "Q",
            }
            for series_key in series_keys
        ]

        snapshot_samples = _time_call(
            lambda: pit.snapshot_ref(snapshot_query),
            iterations=iterations,
        )
        panel_samples = _time_call(
            lambda: pit.build_snapshot_panel_long(
                panel_specs,
                asof=asof,
                align="quarter_end",
            ),
            iterations=iterations,
        )

        return {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "iterations": iterations,
            "periods": periods,
            "series_count": series_count,
            "revisions_per_period": revisions_per_period,
            "rows_loaded": int(len(observations)),
            "snapshot_ref_median_ms": round(statistics.median(snapshot_samples), 3),
            "snapshot_panel_long_median_ms": round(statistics.median(panel_samples), 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT contract benchmarks.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--series-count", type=int, default=3)
    parser.add_argument("--revisions-per-period", type=int, default=2)
    args = parser.parse_args()

    metrics = run_pit_contract_benchmarks(
        iterations=args.iterations,
        periods=args.periods,
        series_count=args.series_count,
        revisions_per_period=args.revisions_per_period,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
