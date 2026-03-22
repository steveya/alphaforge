"""Tests for PIT panel builder and vintage utilities."""
from __future__ import annotations

from datetime import date

import pandas as pd

from alphaforge.pit.vintage import select_vintage_for_asof, validate_no_lookahead


class TestSelectVintageForAsof:
    def test_exact_match(self) -> None:
        vintages = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
        result = select_vintage_for_asof(vintages, date(2025, 2, 1))
        assert result == date(2025, 2, 1)

    def test_between_vintages(self) -> None:
        vintages = [date(2025, 1, 1), date(2025, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2025, 1, 15))
        assert result == date(2025, 1, 1)

    def test_before_all_vintages(self) -> None:
        vintages = [date(2025, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2025, 1, 1))
        assert result is None

    def test_empty_vintages(self) -> None:
        assert select_vintage_for_asof([], date(2025, 1, 1)) is None

    def test_after_all_vintages(self) -> None:
        vintages = [date(2025, 1, 1), date(2025, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2025, 6, 1))
        assert result == date(2025, 2, 1)


class TestValidateNoLookahead:
    def test_valid(self) -> None:
        assert validate_no_lookahead(date(2025, 1, 1), date(2025, 2, 1)) is True

    def test_invalid(self) -> None:
        assert validate_no_lookahead(date(2025, 3, 1), date(2025, 2, 1)) is False

    def test_equal(self) -> None:
        assert validate_no_lookahead(date(2025, 1, 1), date(2025, 1, 1)) is True


class TestBuildPitPanel:
    def test_build_pit_panel_basic(self, tmp_path) -> None:
        from alphaforge.pit.accessor import PITAccessor, ensure_pit_table
        from alphaforge.pit.panel import build_pit_panel
        from alphaforge.store.duckdb_parquet import DuckDBParquetStore

        store = DuckDBParquetStore(root=str(tmp_path / "store"))
        conn = store.conn()
        ensure_pit_table(conn)
        pit = PITAccessor(conn)

        # Insert some test data
        obs = pd.DataFrame([
            {"series_key": "test.a", "obs_date": pd.Timestamp("2025-01-01"), "asof_utc": pd.Timestamp("2025-01-02", tz="UTC"), "value": 1.0, "source": "test"},
            {"series_key": "test.a", "obs_date": pd.Timestamp("2025-01-08"), "asof_utc": pd.Timestamp("2025-01-09", tz="UTC"), "value": 2.0, "source": "test"},
            {"series_key": "test.b", "obs_date": pd.Timestamp("2025-01-01"), "asof_utc": pd.Timestamp("2025-01-02", tz="UTC"), "value": 10.0, "source": "test"},
        ])
        pit.upsert_pit_observations(obs, strict="coerce")

        panel = build_pit_panel(
            pit,
            series_keys={"col_a": "test.a", "col_b": "test.b"},
            asof=pd.Timestamp("2025-02-01", tz="UTC"),
        )
        assert "col_a" in panel.columns
        assert "col_b" in panel.columns
        assert len(panel) >= 1


class TestLongToWide:
    def test_basic_pivot(self) -> None:
        from alphaforge.pit.panel import long_to_wide

        df = pd.DataFrame({
            "obs_date": [pd.Timestamp("2025-01-01")] * 2,
            "series_key": ["a", "b"],
            "value": [1.0, 2.0],
        })
        wide = long_to_wide(df)
        assert "a" in wide.columns
        assert "b" in wide.columns
