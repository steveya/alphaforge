from datetime import timedelta

import pandas as pd
import pytest

from alphaforge.data.adapter import SourceAdapterBase
from alphaforge.data.context import DataContext
from alphaforge.data.contracts import (
    DatasetContract,
    FetchPolicy,
    UnsupportedQueryError,
    resolve_fetch_policy,
)
from alphaforge.data.query import Query
from alphaforge.data.types import FetchResult


def market_contract():
    return DatasetContract(columns=frozenset({"close"}), result_kind="observations")


@pytest.mark.parametrize(
    "options",
    [
        {"asof": "2025-01-01"},
        {"vintage": "first"},
        {"vintage": "specific", "vintage_id": "missing"},
        {"vintage_id": "unused"},
        {"grid": "monthly"},
        {"columns": ["bogus"]},
    ],
)
def test_contract_rejects_unsupported_query_before_acquisition(options):
    args = {"table": "market.ohlcv", "columns": ["close"], **options}
    with pytest.raises(UnsupportedQueryError):
        market_contract().validate_query(Query(**args))


def test_contract_distinguishes_no_entity_selector_from_empty_selection():
    contract = DatasetContract(
        columns=frozenset({"value"}),
        result_kind="snapshot",
        requires_entities=True,
        supports_asof=True,
        availability_precision="date",
        availability_basis="snapshot",
    )
    with pytest.raises(UnsupportedQueryError, match="entities"):
        contract.validate_query(Query("macro.fred", ["value"]))
    contract.validate_query(Query("macro.fred", ["value"], entities=[]))


def test_inverted_query_bounds_are_rejected():
    with pytest.raises(ValueError, match="start"):
        Query("x", [], start="2025-02-01", end="2025-01-01")


def test_policy_is_explicit_and_legacy_staleness_is_preserved():
    assert resolve_fetch_policy(None, None) == FetchPolicy()
    assert resolve_fetch_policy(None, timedelta(0)).cache_mode == "refresh"
    assert resolve_fetch_policy(None, timedelta(days=1)) == FetchPolicy(
        cache_mode="allow_stale", max_age=timedelta(days=1)
    )
    with pytest.raises(ValueError, match="combine"):
        resolve_fetch_policy(FetchPolicy(), timedelta(days=1))
    with pytest.raises(ValueError):
        FetchPolicy(max_age=timedelta(seconds=-1))
    with pytest.raises(ValueError):
        FetchPolicy(cache_mode="unknown")


class DeclaredAdapter(SourceAdapterBase):
    source_name = "provider"
    datasets = frozenset({"market.ohlcv"})

    def __init__(self, value):
        self.value = value
        self.calls = []

    def describe(self, dataset):
        return market_contract()

    def fetch(self, query, *, max_staleness=None, policy=None):
        self.calls.append(policy)
        return FetchResult(
            pd.DataFrame(
                {
                    "series_key": ["A"],
                    "obs_date": pd.to_datetime(["2025-01-01"]),
                    "close": [self.value],
                }
            ),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
            result_kind="observations",
        )


def test_named_registration_routes_distinct_instances_and_batch_policy():
    raw, adjusted = DeclaredAdapter(1), DeclaredAdapter(2)
    ctx = DataContext.from_adapters(named_adapters={"raw": raw, "adjusted": adjusted})
    with pytest.raises(KeyError, match="Multiple"):
        ctx.load("market.ohlcv", columns=["close"])
    assert (
        ctx.load("market.ohlcv", columns=["close"], source="raw").data.close.iloc[0]
        == 1
    )
    routed = ctx.load("market.ohlcv", columns=["close"], source="adjusted")
    assert routed.source == "provider"
    assert routed.adapter_name == "adjusted"
    policy = FetchPolicy(cache_mode="refresh")
    results = ctx.fetch_many(
        [Query("market.ohlcv", ["close"])], source="adjusted", policy=policy
    )
    assert results[0].data.close.iloc[0] == 2
    assert adjusted.calls == [None, policy]
    with pytest.raises(ValueError, match="Duplicate"):
        DataContext.from_adapters(raw, adjusted)
    with pytest.raises(ValueError, match="Duplicate"):
        DataContext.from_adapters(raw, named_adapters={"provider": adjusted})


def test_context_validates_declared_queries_and_results():
    adapter = DeclaredAdapter(1)
    ctx = DataContext.from_adapters(adapter)
    with pytest.raises(UnsupportedQueryError):
        ctx.load("market.ohlcv", columns=["close"], vintage="first")
    assert adapter.calls == []
    # A conforming signature cannot conceal a malformed returned frame.
    original = adapter.fetch

    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.data.drop(columns="close", inplace=True)
        return result

    adapter.fetch = malformed
    with pytest.raises(ValueError, match="close"):
        ctx.load("market.ohlcv", columns=["close"])


@pytest.mark.parametrize("snapshot_asof", [None, "2025-01-01", "2025-01-03"])
@pytest.mark.parametrize("empty", [False, True])
def test_snapshot_contract_checks_cutoff_metadata_even_when_empty(snapshot_asof, empty):
    contract = DatasetContract(
        columns=frozenset({"value", "asof_utc"}),
        result_kind="snapshot",
        availability_precision="date",
        availability_basis="snapshot",
        supports_asof=True,
    )
    frame = pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2024-10-01")],
            "asof_utc": [pd.Timestamp("2025-01-01", tz="UTC")],
            "value": [1.0],
        }
    )
    result = FetchResult(
        frame.iloc[:0] if empty else frame,
        "fred",
        "macro.fred",
        True,
        None,
        result_kind="snapshot",
        availability_precision="date",
        availability_basis="snapshot",
        snapshot_asof=snapshot_asof,
    )
    with pytest.raises(ValueError, match="snapshot"):
        contract.validate_result(
            Query("macro.fred", ["value"], asof="2025-01-02"), result
        )


def test_date_snapshot_contract_accepts_the_requested_utc_day():
    contract = DatasetContract(
        columns=frozenset({"value", "asof_utc"}),
        result_kind="snapshot",
        availability_precision="date",
        availability_basis="snapshot",
        supports_asof=True,
    )
    result = FetchResult(
        pd.DataFrame(
            {
                "series_key": ["GDP"],
                "obs_date": [pd.Timestamp("2024-10-01")],
                "asof_utc": [pd.Timestamp("2025-01-02T18:00Z")],
                "value": [1.0],
            }
        ),
        "fred",
        "macro.fred",
        True,
        None,
        result_kind="snapshot",
        availability_precision="date",
        availability_basis="snapshot",
        snapshot_asof=pd.Timestamp("2025-01-02", tz="UTC"),
    )
    contract.validate_result(
        Query("macro.fred", ["value"], asof="2025-01-01T20:00-05:00"), result
    )


def test_empty_column_selection_requires_the_entire_declared_schema():
    result = FetchResult(
        pd.DataFrame(columns=["series_key", "obs_date"]),
        "provider",
        "market.ohlcv",
        False,
        None,
        result_kind="observations",
    )
    with pytest.raises(ValueError, match="close"):
        market_contract().validate_result(Query("market.ohlcv", []), result)


@pytest.mark.parametrize("precision", ["date", "timestamp"])
def test_snapshot_contract_rejects_rows_after_selected_cutoff_without_query_asof(
    precision,
):
    contract = DatasetContract(
        columns=frozenset({"value", "asof_utc"}),
        result_kind="snapshot",
        availability_precision=precision,
        availability_basis="snapshot",
        supports_asof=True,
    )
    result = FetchResult(
        pd.DataFrame(
            {
                "series_key": ["GDP"],
                "obs_date": [pd.Timestamp("2024-10-01", tz="UTC")],
                "asof_utc": [pd.Timestamp("2025-01-03", tz="UTC")],
                "value": [1.0],
            }
        ),
        "fred",
        "macro.fred",
        True,
        None,
        result_kind="snapshot",
        availability_precision=precision,
        availability_basis="snapshot",
        snapshot_asof=pd.Timestamp("2025-01-02", tz="UTC"),
    )

    with pytest.raises(ValueError, match="snapshot cutoff"):
        contract.validate_result(Query("macro.fred", ["value"]), result)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "policy", [None, FetchPolicy(), FetchPolicy(allow_partial=True)]
)
def test_context_requires_explicit_partial_result_permission(batch, policy):
    from dataclasses import replace

    from alphaforge.data.types import FetchFailure

    class PartialAdapter(DeclaredAdapter):
        def fetch(self, query, *, max_staleness=None, policy=None):
            result = super().fetch(query, max_staleness=max_staleness, policy=policy)
            return replace(
                result, failures=(FetchFailure(entity="B", message="unavailable"),)
            )

    ctx = DataContext.from_adapters(PartialAdapter(1))
    query = Query("market.ohlcv", ["close"])
    fetch = ctx.fetch_many if batch else ctx.fetch
    request = [query] if batch else query
    if policy is None or not policy.allow_partial:
        with pytest.raises(ValueError, match="allow_partial"):
            fetch(request, policy=policy)
    else:
        result = fetch(request, policy=policy)
        result = result[0] if batch else result
        assert not result.complete
        assert result.failures[0].entity == "B"
        assert result.adapter_name == "provider"


@pytest.mark.parametrize("precision", ["date", "timestamp"])
def test_revision_cutoff_respects_availability_precision(precision):
    contract = DatasetContract(
        columns=frozenset({"value"}), result_kind="revisions",
        availability_precision=precision, availability_basis="release", supports_asof=True,
    )
    result = FetchResult(
        pd.DataFrame({"series_key": ["GDP"], "obs_date": [pd.Timestamp("2024-10-01")],
                      "asof_utc": [pd.Timestamp("2025-01-03T15:00Z")], "value": [1.0]}),
        "fred", "macro.fred", True, None, result_kind="revisions",
        availability_precision=precision, availability_basis="release",
    )
    query = Query("macro.fred", ["value"], asof="2025-01-03")
    if precision == "date":
        contract.validate_result(query, result)
        assert result.as_timeseries(asof=query.asof).tolist() == [1.0]
    else:
        with pytest.raises(ValueError, match="asof cutoff"):
            contract.validate_result(query, result)
        assert result.as_timeseries(asof=query.asof).empty


@pytest.mark.parametrize("result_kind", ["snapshot", "revisions"])
def test_availability_identity_column_is_queryable_without_explicit_declaration(result_kind):
    contract = DatasetContract(columns=frozenset({"value"}), result_kind=result_kind)
    contract.validate_query(Query("macro.fred", ["value", "asof_utc"]))


@pytest.mark.parametrize("extra_column", ["open", "undeclared"])
@pytest.mark.parametrize("empty", [False, True])
def test_contract_rejects_columns_outside_requested_projection(extra_column, empty):
    contract = DatasetContract(columns=frozenset({"close", "open"}), result_kind="observations")
    frame = pd.DataFrame({"series_key": ["A"], "obs_date": [pd.Timestamp("2025-01-01")],
                          "close": [1.0], extra_column: [2.0]})
    result = FetchResult(frame.iloc[:0] if empty else frame, "provider", "market.ohlcv",
                         False, None, result_kind="observations")
    with pytest.raises(ValueError, match="columns"):
        contract.validate_result(Query("market.ohlcv", ["close"]), result)
