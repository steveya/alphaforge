"""Tests for alphaforge.pipeline protocols and SimplePipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from alphaforge.pipeline.protocols import (
    Filter,
    Parametric,
    PipelineVariant,
    Signal,
    SimplePipeline,
    Transformer,
)

# --- Stub implementations ---


@dataclass
class StubFilter:
    name: str = "stub_filter"
    keep_above: float = 0.0

    def apply(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        return df[df["value"] > self.keep_above].copy()


@dataclass
class StubTransformer:
    name: str = "stub_transformer"
    multiplier: float = 2.0

    def transform(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        out = df.copy()
        out["value"] = out["value"] * self.multiplier
        return out

    def get_params(self) -> dict[str, float]:
        return {"multiplier": self.multiplier}

    def set_params(self, params: dict[str, float]) -> None:
        if "multiplier" in params:
            self.multiplier = params["multiplier"]


@dataclass
class StubSignal:
    name: str = "stub_signal"

    def score(self, df: pd.DataFrame, **params: Any) -> pd.Series:
        return df.groupby("entity_id")["value"].last().rename(self.name)


@dataclass
class StubRemoveAllFilter:
    name: str = "remove_all"

    def apply(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        return df.iloc[0:0].copy()


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": ["a", "a", "b", "b", "c", "c"],
        "date": pd.date_range("2026-01-01", periods=6, freq="D"),
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


# --- Tests ---


class TestFilterProtocol:
    def test_filter_protocol(self) -> None:
        f = StubFilter()
        assert isinstance(f, Filter)


class TestTransformerProtocol:
    def test_transformer_protocol(self) -> None:
        t = StubTransformer()
        assert isinstance(t, Transformer)


class TestSignalProtocol:
    def test_signal_protocol(self) -> None:
        s = StubSignal()
        assert isinstance(s, Signal)


class TestParametricProtocol:
    def test_parametric_protocol(self) -> None:
        t = StubTransformer()
        assert isinstance(t, Parametric)


class TestSimplePipelineChain:
    def test_simple_pipeline_chain(self) -> None:
        pipe = SimplePipeline(
            name="test",
            filters=[StubFilter(keep_above=2.0)],
            transformers=[StubTransformer(multiplier=10.0)],
            signal=StubSignal(),
        )
        df = _make_df()
        result = pipe.run(df)
        assert not result.empty
        # After filter: values > 2.0 → {3,4,5,6}, after mult×10 → {30,40,50,60}
        # Last per entity: b→40, c→60
        assert result["b"] == 40.0
        assert result["c"] == 60.0


class TestSimplePipelineNoSignalRaises:
    def test_simple_pipeline_no_signal_raises(self) -> None:
        pipe = SimplePipeline(name="no_signal")
        with pytest.raises(ValueError, match="no terminal Signal"):
            pipe.run(_make_df())


class TestSimplePipelineGetSetParams:
    def test_simple_pipeline_get_set_params(self) -> None:
        t = StubTransformer(multiplier=5.0)
        pipe = SimplePipeline(name="test", transformers=[t], signal=StubSignal())

        params = pipe.get_all_params()
        assert params == {"stub_transformer": {"multiplier": 5.0}}

        pipe.set_all_params({"stub_transformer": {"multiplier": 3.0}})
        assert t.multiplier == 3.0


class TestSimplePipelineEmptyAfterFilter:
    def test_simple_pipeline_empty_after_filter(self) -> None:
        pipe = SimplePipeline(
            name="test",
            filters=[StubRemoveAllFilter()],
            transformers=[StubTransformer()],
            signal=StubSignal(),
        )
        result = pipe.run(_make_df())
        assert result.empty


class TestPipelineVariant:
    def test_pipeline_variant(self) -> None:
        pipe = SimplePipeline(
            name="inner",
            transformers=[StubTransformer(multiplier=1.0)],
            signal=StubSignal(),
        )
        variant = PipelineVariant(name="v1", pipeline=pipe, description="test variant")
        result = variant.pipeline.run(_make_df())
        assert not result.empty
        assert variant.name == "v1"
