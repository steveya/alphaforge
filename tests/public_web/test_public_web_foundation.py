from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data.public_web.archive import (
    ArchiveFetchPlanEntry,
    discover_archive_fetches,
    discover_archive_links,
    filter_urls_for_years,
    iter_yearly_archive_fetches,
    iter_yearly_archive_urls,
)
from alphaforge.data.public_web.base import PublicWebSourceBase
from alphaforge.data.public_web.finalize import (
    empty_frame_for_schema,
    finalize_public_frame,
)
from alphaforge.data.public_web.registry_api import RegistryApiSourceBase
from alphaforge.data.public_web.schema_helpers import (
    event_table_schema,
    single_value_schema,
)
from alphaforge.data.public_web.tabular import (
    artifact_name_from_url,
    candidate_tables,
    resolved_date_series,
    resolved_text_series,
)
from alphaforge.data.query import Query


class _DummySource(PublicWebSourceBase):
    name = "dummy"
    TABLE = "dummy_series"

    def schemas(self):
        return {self.TABLE: single_value_schema(self.TABLE)}


class _DummyRegistrySource(RegistryApiSourceBase):
    name = "dummy_registry"
    TABLE = "dummy_registry_series"

    def __init__(self, *, registry_entries=None) -> None:
        super().__init__(http_client=None)
        self._init_registry(
            "dummy_registry.yaml",
            registry_entries=registry_entries,
            registry_path=None,
        )

    def schemas(self):
        return {self.TABLE: single_value_schema(self.TABLE)}


def test_empty_frame_for_schema_uses_schema_columns() -> None:
    schema = single_value_schema("dummy_series")

    out = empty_frame_for_schema(schema)

    assert list(out.columns) == ["date", "entity_id", "asof_utc", "value"]
    assert out.empty


def test_event_table_schema_preserves_event_metadata() -> None:
    schema = event_table_schema(
        "dummy.events",
        required_columns=["value"],
        native_freq="D",
        time_semantics="point",
        expected_cadence_days=1,
    )

    assert schema.time_column == "ts_utc"
    assert schema.event_time_column == "ts_utc"
    assert schema.native_freq == "D"
    assert schema.time_semantics == "point"
    assert schema.expected_cadence_days == 1


def test_finalize_public_frame_filters_projects_and_sorts() -> None:
    schema = single_value_schema("dummy_series")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-01-31"], utc=True),
            "entity_id": ["series_b", "series_a"],
            "asof_utc": pd.to_datetime(["2020-03-02", "2020-03-01"], utc=True),
            "value": [2.0, 1.0],
            "ignored": [9, 8],
        }
    )

    out = finalize_public_frame(
        frame,
        q=Query(
            table="dummy_series",
            columns=["value"],
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-12-31"),
            entities=["series_a", "series_b"],
            asof=pd.Timestamp("2020-03-02"),
        ),
        schema=schema,
    )

    assert list(out.columns) == ["date", "entity_id", "asof_utc", "value"]
    assert out["entity_id"].tolist() == ["series_a", "series_b"]
    assert out["value"].tolist() == [1.0, 2.0]


def test_public_web_source_base_builds_schema_empty_frame_from_records() -> None:
    src = _DummySource()
    schema = src.schemas()[src.TABLE]

    out = src._frame_from_records([], schema=schema)

    assert list(out.columns) == ["date", "entity_id", "asof_utc", "value"]
    assert out.empty


def test_registry_api_source_iter_entity_configs_requires_entities() -> None:
    src = _DummyRegistrySource(registry_entries=[{"entity_id": "TEST", "foo": "bar"}])

    with pytest.raises(ValueError, match="requires entities"):
        list(
            src._iter_entity_configs(
                Query(table=src.TABLE, columns=["value"]),
                error_message="dummy registry requires entities",
            )
        )


def test_registry_api_source_iter_entity_configs_skips_unknown_entities() -> None:
    src = _DummyRegistrySource(registry_entries=[{"entity_id": "TEST", "foo": "bar"}])

    pairs = list(
        src._iter_entity_configs(
            Query(table=src.TABLE, columns=["value"], entities=["TEST", "MISSING"]),
            error_message="dummy registry requires entities",
        )
    )

    assert len(pairs) == 1
    assert pairs[0][0] == "TEST"
    assert pairs[0][1]["foo"] == "bar"


def test_tabular_candidate_tables_filters_by_any_and_all_columns() -> None:
    matching = pd.DataFrame({"volume": [1], "date": ["2020-01-01"]})
    missing = pd.DataFrame({"price": [1]})

    out = candidate_tables(
        [matching, missing],
        any_of=["volume", "open_interest"],
        all_of=["date"],
    )

    assert out == [matching]


def test_tabular_resolved_helpers_apply_defaults_and_normalization() -> None:
    frame = pd.DataFrame({"trading_day": ["2020-01-02"], "group": ["Equity Index"]})

    dates = resolved_date_series(
        frame,
        ["date", "trading_day"],
        default_date=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    groups = resolved_text_series(
        frame,
        ["product_group", "group"],
        default="unknown",
        case="lower",
        space_replacement="_",
    )

    assert str(dates.iloc[0]) == "2020-01-02 00:00:00+00:00"
    assert groups.iloc[0] == "equity_index"


def test_tabular_artifact_name_from_url_strips_query_string() -> None:
    assert artifact_name_from_url("https://example.com/path/data.csv?x=1", "fallback") == "data.csv"


def test_archive_helpers_discover_filter_and_plan_urls() -> None:
    html = (
        '<a href="/files/report_2020.zip?download=1">a</a>'
        '<a href="/files/report_2021.zip">b</a>'
    )
    links = discover_archive_links(
        html,
        base_url="https://example.com/archive/index.html",
        suffixes=[".zip"],
    )

    filtered = filter_urls_for_years(links, [2021])
    planned = discover_archive_fetches(
        html,
        base_url="https://example.com/archive/index.html",
        suffixes=[".zip"],
        years=[2021],
    )
    yearly = iter_yearly_archive_urls(
        start_year=2009,
        end_year=2012,
        url_template="https://example.com/data_{year}.zip",
        first_year=2006,
        yearly_first_year=2010,
        historical_url="https://example.com/historical.zip",
        historical_last_year=2011,
    )
    yearly_planned = iter_yearly_archive_fetches(
        start_year=2009,
        end_year=2012,
        url_template="https://example.com/data_{year}.zip",
        first_year=2006,
        yearly_first_year=2010,
        historical_url="https://example.com/historical.zip",
        historical_last_year=2011,
    )

    assert links == [
        "https://example.com/files/report_2020.zip?download=1",
        "https://example.com/files/report_2021.zip",
    ]
    assert filtered == ["https://example.com/files/report_2021.zip"]
    assert planned == [
        ArchiveFetchPlanEntry(
            url="https://example.com/files/report_2021.zip",
            artifact_name="report_2021.zip",
            year=2021,
        )
    ]
    assert yearly == [
        "https://example.com/historical.zip",
        "https://example.com/data_2012.zip",
    ]
    assert yearly_planned == [
        ArchiveFetchPlanEntry(
            url="https://example.com/historical.zip",
            artifact_name="historical.zip",
            year=None,
        ),
        ArchiveFetchPlanEntry(
            url="https://example.com/data_2012.zip",
            artifact_name="data_2012.zip",
            year=2012,
        ),
    ]
