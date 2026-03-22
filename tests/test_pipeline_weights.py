"""Tests for pipeline weight adjustment."""
from __future__ import annotations

from alphaforge.pipeline.weights import adjust_weights_for_health


class TestAdjustWeightsForHealth:
    def test_no_degradation(self) -> None:
        weights = {"a": 0.5, "b": 0.5}
        result = adjust_weights_for_health(
            weights=weights,
            health_fn=lambda _: 1.0,
            source_map={"a": "src_a", "b": "src_b"},
        )
        assert result == {"a": 0.5, "b": 0.5}

    def test_partial_degradation_renormalized(self) -> None:
        weights = {"a": 0.5, "b": 0.5}
        result = adjust_weights_for_health(
            weights=weights,
            health_fn=lambda s: 0.5 if s == "src_a" else 1.0,
            source_map={"a": "src_a", "b": "src_b"},
        )
        # a: 0.5*0.5 = 0.25, b: 0.5*1.0 = 0.5, total = 0.75
        # renormalized: a = 0.25 * (1.0/0.75), b = 0.5 * (1.0/0.75)
        assert abs(result["a"] + result["b"] - 1.0) < 1e-10

    def test_dead_source_gets_zero(self) -> None:
        weights = {"a": 0.5, "b": 0.5}
        result = adjust_weights_for_health(
            weights=weights,
            health_fn=lambda s: 0.0 if s == "src_a" else 1.0,
            source_map={"a": "src_a", "b": "src_b"},
        )
        assert result["a"] == 0.0
        assert abs(result["b"] - 1.0) < 1e-10
