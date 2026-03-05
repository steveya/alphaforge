from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, cast

import pandas as pd

from .exceptions import PITContractError

JoinMode = Literal["inner", "left", "right", "outer"]


class ReleaseRankPolicy(TypedDict):
    mode: Literal["rank"]
    rank: int


class ReleaseHorizonPolicy(TypedDict):
    mode: Literal["horizon"]
    horizon: pd.Timedelta | str


ReleaseSelectionPolicy = Literal["first", "latest"] | ReleaseRankPolicy | ReleaseHorizonPolicy


@dataclass(frozen=True)
class ReleaseRecord:
    series_key: str
    ref_key: str
    obs_date: pd.Timestamp
    asof_utc: pd.Timestamp
    release_rank: int
    value: float | None
    revision_id: str | None


@dataclass(frozen=True)
class PITExpressionNode:
    name: str
    output_series_key: str
    expression: str
    inputs: Mapping[str, str]
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    join: JoinMode = "inner"
    fill_value: float | None = None

    def node_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_series_key": self.output_series_key,
            "expression": self.expression,
            "inputs": dict(self.inputs),
            "depends_on": list(self.depends_on),
            "join": self.join,
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class PITExpressionGraphSpec:
    nodes: tuple[PITExpressionNode, ...]
    graph_id: str | None = None
    description: str | None = None

    def _validate(self) -> None:
        if not self.nodes:
            raise PITContractError("Expression graph requires at least one node.")

        names = [node.name for node in self.nodes]
        if any(not n.strip() for n in names):
            raise PITContractError("Every expression node must have a non-empty name.")
        if len(names) != len(set(names)):
            raise PITContractError("Expression node names must be unique.")

        known = set(names)
        for node in self.nodes:
            if not node.output_series_key.strip():
                raise PITContractError(f"Node '{node.name}' requires output_series_key.")
            if not node.expression.strip():
                raise PITContractError(f"Node '{node.name}' requires expression.")
            if node.join not in {"inner", "left", "right", "outer"}:
                raise PITContractError(
                    f"Node '{node.name}' has unsupported join='{node.join}'."
                )
            if not node.inputs:
                raise PITContractError(f"Node '{node.name}' requires at least one input alias.")
            for alias, series_key in node.inputs.items():
                if not str(alias).strip():
                    raise PITContractError(f"Node '{node.name}' has empty input alias.")
                if not str(series_key).strip():
                    raise PITContractError(
                        f"Node '{node.name}' has empty series key for alias '{alias}'."
                    )
            for dep in node.depends_on:
                if dep not in known:
                    raise PITContractError(
                        f"Node '{node.name}' depends on unknown node '{dep}'."
                    )

    def ordered_nodes(self) -> tuple[PITExpressionNode, ...]:
        self._validate()

        pending = {node.name: node for node in self.nodes}
        declared_order = [node.name for node in self.nodes]
        resolved: list[PITExpressionNode] = []
        done: set[str] = set()

        while pending:
            ready_names = [
                name
                for name in declared_order
                if name in pending and all(dep in done for dep in pending[name].depends_on)
            ]
            if not ready_names:
                unresolved = ", ".join(sorted(pending))
                raise PITContractError(
                    "Expression graph contains a cycle or unresolved dependencies: "
                    f"{unresolved}"
                )

            for name in ready_names:
                node = pending[name]
                resolved.append(node)
                done.add(node.name)
                pending.pop(node.name, None)

        return tuple(resolved)

    def spec_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "description": self.description,
            "nodes": [node.node_payload() for node in self.ordered_nodes()],
        }

    def spec_hash(self) -> str:
        payload = json.dumps(self.spec_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolved_graph_id(self) -> str:
        if self.graph_id is not None and self.graph_id.strip():
            return self.graph_id.strip()
        return f"pit_expr_graph:{self.spec_hash()[:16]}"


@dataclass(frozen=True)
class PITExpressionGraphResult:
    graph_id: str
    run_id: str
    status: str
    rows_written: int
    node_rows_written: dict[str, int]
    run_started_utc: pd.Timestamp
    run_finished_utc: pd.Timestamp
    incremental: bool
    effective_start_asof: pd.Timestamp | None = None


@dataclass(frozen=True)
class SnapshotSeriesSpec:
    series_key: str
    alias: str | None = None
    start_ref: str | None = None
    end_ref: str | None = None
    release_policy: ReleaseSelectionPolicy = "latest"


def _cast_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def coerce_expression_graph_spec(
    spec: PITExpressionGraphSpec | Mapping[str, Any],
) -> PITExpressionGraphSpec:
    if isinstance(spec, PITExpressionGraphSpec):
        spec._validate()
        return spec

    if not isinstance(spec, Mapping):
        raise PITContractError(
            "Expression graph spec must be PITExpressionGraphSpec or a mapping."
        )

    allowed_graph_keys = {"nodes", "graph_id", "description"}
    unknown_graph_keys = sorted(set(spec.keys()) - allowed_graph_keys)
    if unknown_graph_keys:
        raise PITContractError(f"Unknown expression graph spec keys: {unknown_graph_keys}")

    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, (list, tuple)):
        raise PITContractError("Expression graph spec requires 'nodes' as a list/tuple.")

    nodes: list[PITExpressionNode] = []
    for idx, raw in enumerate(raw_nodes):
        if isinstance(raw, PITExpressionNode):
            nodes.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise PITContractError(
                f"Expression node {idx} must be mapping or PITExpressionNode."
            )

        allowed_node_keys = {
            "name",
            "output_series_key",
            "expression",
            "inputs",
            "depends_on",
            "join",
            "fill_value",
        }
        unknown_node_keys = sorted(set(raw.keys()) - allowed_node_keys)
        if unknown_node_keys:
            raise PITContractError(
                f"Unknown expression node keys at index {idx}: {unknown_node_keys}"
            )

        name = str(raw.get("name", "")).strip()
        if not name:
            raise PITContractError(f"Expression node {idx} requires non-empty 'name'.")

        output_series_key = str(raw.get("output_series_key", "")).strip()
        expression = str(raw.get("expression", "")).strip()
        raw_inputs = raw.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise PITContractError(f"Expression node '{name}' requires 'inputs' mapping.")

        inputs = {str(k): str(v) for k, v in raw_inputs.items()}

        raw_depends_on = raw.get("depends_on", ())
        depends_on: tuple[str, ...]
        if isinstance(raw_depends_on, str):
            depends_on = (raw_depends_on,)
        elif isinstance(raw_depends_on, (list, tuple)):
            depends_on = tuple(str(dep) for dep in raw_depends_on)
        else:
            raise PITContractError(
                f"Expression node '{name}' has invalid depends_on type: "
                f"{type(raw_depends_on).__name__}."
            )

        join = str(raw.get("join", "inner")).strip().lower()
        fill_value = raw.get("fill_value")
        if fill_value is not None:
            fill_value = float(fill_value)

        nodes.append(
            PITExpressionNode(
                name=name,
                output_series_key=output_series_key,
                expression=expression,
                inputs=inputs,
                depends_on=depends_on,
                join=join,  # type: ignore[arg-type]
                fill_value=fill_value,
            )
        )

    graph = PITExpressionGraphSpec(
        nodes=tuple(nodes),
        graph_id=_cast_or_none(spec.get("graph_id")),
        description=_cast_or_none(spec.get("description")),
    )
    graph._validate()
    return graph


def coerce_snapshot_series_spec(
    spec: SnapshotSeriesSpec | Mapping[str, Any],
) -> SnapshotSeriesSpec:
    if isinstance(spec, SnapshotSeriesSpec):
        return spec

    if not isinstance(spec, Mapping):
        raise PITContractError("Snapshot series spec must be SnapshotSeriesSpec or a mapping.")

    allowed_keys = {"series_key", "alias", "start_ref", "end_ref", "release_policy"}
    unknown_keys = sorted(set(spec.keys()) - allowed_keys)
    if unknown_keys:
        raise PITContractError(f"Unknown snapshot series spec keys: {unknown_keys}")

    series_key = str(spec.get("series_key", "")).strip()
    if not series_key:
        raise PITContractError("Snapshot series spec requires non-empty series_key.")

    alias = _cast_or_none(spec.get("alias"))
    start_ref = _cast_or_none(spec.get("start_ref"))
    end_ref = _cast_or_none(spec.get("end_ref"))
    release_policy = spec.get("release_policy", "latest")

    return SnapshotSeriesSpec(
        series_key=series_key,
        alias=alias,
        start_ref=start_ref,
        end_ref=end_ref,
        release_policy=release_policy,
    )


def normalize_release_selection_policy(
    policy: ReleaseSelectionPolicy | Mapping[str, Any] | str,
) -> tuple[str, int | pd.Timedelta | None]:
    if isinstance(policy, str):
        mode = policy.strip().lower()
        if mode not in {"first", "latest"}:
            raise PITContractError(
                "Release policy must be 'first', 'latest', {'mode':'rank',...}, or {'mode':'horizon',...}."
            )
        return mode, None

    if not isinstance(policy, Mapping):
        raise PITContractError("Release policy must be a string or mapping.")

    policy_map: Mapping[str, Any] = cast(Mapping[str, Any], policy)
    mode = str(policy_map.get("mode", "")).strip().lower()
    if mode == "rank":
        if "rank" not in policy_map:
            raise PITContractError("Release policy mode='rank' requires 'rank'.")
        rank = int(policy_map["rank"])
        if rank <= 0:
            raise PITContractError("Release policy rank must be > 0.")
        return mode, rank

    if mode == "horizon":
        if "horizon" not in policy_map:
            raise PITContractError("Release policy mode='horizon' requires 'horizon'.")
        horizon = pd.Timedelta(policy_map["horizon"])
        if horizon < pd.Timedelta(0):
            raise PITContractError("Release policy horizon must be >= 0.")
        return mode, horizon

    raise PITContractError(
        "Release policy must be 'first', 'latest', {'mode':'rank',...}, or {'mode':'horizon',...}."
    )
