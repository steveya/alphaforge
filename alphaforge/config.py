"""Generic configuration resolution utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass


def resolve_config_value(
    explicit: str | None,
    env_var: str,
    default: str,
) -> str:
    """Resolve from explicit → env → default."""
    return explicit or os.environ.get(env_var, default)


@dataclass(frozen=True)
class ConfigEntry:
    """A single resolvable configuration value."""

    name: str
    env_var: str
    default: str
    description: str = ""

    def resolve(self, explicit: str | None = None) -> str:
        return resolve_config_value(explicit, self.env_var, self.default)
