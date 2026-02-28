from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from .exceptions import PITContractError
from .transforms import PITTransformResult, PITTransformSpec, coerce_transform_spec


@dataclass(frozen=True)
class PITPipelineStep:
    name: str
    spec: PITTransformSpec | Mapping[str, Any]
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def normalized_spec(self) -> PITTransformSpec:
        return coerce_transform_spec(self.spec)

    def step_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depends_on": list(self.depends_on),
            "spec": self.normalized_spec().spec_payload(),
        }


@dataclass(frozen=True)
class PITPipelineSpec:
    steps: tuple[PITPipelineStep, ...]
    pipeline_id: str | None = None
    description: str | None = None

    def _validate(self) -> None:
        if not self.steps:
            raise PITContractError("PIT pipeline requires at least one step.")

        names = [step.name for step in self.steps]
        if any(not n.strip() for n in names):
            raise PITContractError("Every pipeline step must have a non-empty name.")
        if len(names) != len(set(names)):
            raise PITContractError("Pipeline step names must be unique.")

        known = set(names)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known:
                    raise PITContractError(
                        f"Step '{step.name}' depends on unknown step '{dep}'."
                    )

    def ordered_steps(self) -> tuple[PITPipelineStep, ...]:
        self._validate()

        pending = {step.name: step for step in self.steps}
        declared_order = [step.name for step in self.steps]
        resolved: list[PITPipelineStep] = []
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
                    "Pipeline dependency graph contains a cycle or unresolved dependencies: "
                    f"{unresolved}"
                )

            for name in ready_names:
                step = pending[name]
                resolved.append(step)
                done.add(step.name)
                pending.pop(step.name, None)

        return tuple(resolved)

    def spec_payload(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "description": self.description,
            "steps": [step.step_payload() for step in self.ordered_steps()],
        }

    def spec_hash(self) -> str:
        payload = json.dumps(self.spec_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolved_pipeline_id(self) -> str:
        if self.pipeline_id is not None and self.pipeline_id.strip():
            return self.pipeline_id.strip()
        return f"pit_pipeline:{self.spec_hash()[:16]}"


@dataclass(frozen=True)
class PITPipelineResult:
    pipeline_id: str
    run_id: str
    status: str
    step_results: tuple[PITTransformResult, ...]
    rows_written: int
    run_started_utc: pd.Timestamp
    run_finished_utc: pd.Timestamp
    incremental: bool
    effective_start_asof: pd.Timestamp | None = None


def coerce_pipeline_spec(spec: PITPipelineSpec | Mapping[str, Any]) -> PITPipelineSpec:
    if isinstance(spec, PITPipelineSpec):
        spec._validate()
        return spec

    if not isinstance(spec, Mapping):
        raise PITContractError(
            "Pipeline spec must be PITPipelineSpec or a mapping with pipeline fields."
        )

    allowed_keys = {"steps", "pipeline_id", "description"}
    unknown = sorted(set(spec.keys()) - allowed_keys)
    if unknown:
        raise PITContractError(f"Unknown pipeline spec keys: {unknown}")

    raw_steps = spec.get("steps")
    if not isinstance(raw_steps, (list, tuple)):
        raise PITContractError("Pipeline spec requires 'steps' as a list/tuple.")

    steps: list[PITPipelineStep] = []
    for idx, raw in enumerate(raw_steps):
        if isinstance(raw, PITPipelineStep):
            steps.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise PITContractError(f"Pipeline step {idx} must be mapping or PITPipelineStep.")

        allowed_step_keys = {"name", "spec", "depends_on"}
        unknown_step = sorted(set(raw.keys()) - allowed_step_keys)
        if unknown_step:
            raise PITContractError(f"Unknown pipeline step keys at index {idx}: {unknown_step}")

        name = str(raw.get("name", "")).strip()
        if not name:
            raise PITContractError(f"Pipeline step {idx} requires non-empty 'name'.")
        if "spec" not in raw:
            raise PITContractError(f"Pipeline step '{name}' requires 'spec'.")
        step_spec = coerce_transform_spec(raw["spec"])

        raw_deps = raw.get("depends_on", ())
        deps: tuple[str, ...]
        if isinstance(raw_deps, str):
            deps = (raw_deps,)
        elif isinstance(raw_deps, (list, tuple)):
            deps = tuple(str(dep) for dep in raw_deps)
        else:
            raise PITContractError(
                f"Pipeline step '{name}' has invalid depends_on type: {type(raw_deps).__name__}."
            )

        steps.append(PITPipelineStep(name=name, spec=step_spec, depends_on=deps))

    pipeline = PITPipelineSpec(
        steps=tuple(steps),
        pipeline_id=cast_or_none(spec.get("pipeline_id")),
        description=cast_or_none(spec.get("description")),
    )
    pipeline._validate()
    return pipeline


def cast_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
