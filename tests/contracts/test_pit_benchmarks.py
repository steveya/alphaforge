from __future__ import annotations

from benchmarks.pit import run_pit_contract_benchmarks


def test_pit_benchmark_harness_returns_named_metrics() -> None:
    metrics = run_pit_contract_benchmarks(
        iterations=1,
        periods=8,
        series_count=2,
        revisions_per_period=2,
    )

    assert metrics["rows_loaded"] > 0
    assert metrics["snapshot_ref_median_ms"] >= 0.0
    assert metrics["snapshot_panel_long_median_ms"] >= 0.0
