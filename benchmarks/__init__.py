"""Benchmark harnesses for core-platform regression tracking."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_pit_contract_benchmarks"]


def __getattr__(name: str) -> Any:
    if name == "run_pit_contract_benchmarks":
        return import_module(".pit", __name__).run_pit_contract_benchmarks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
