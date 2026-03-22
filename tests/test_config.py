"""Tests for alphaforge config resolution."""
from __future__ import annotations

from alphaforge.config import ConfigEntry, resolve_config_value


class TestResolveConfigValue:
    def test_explicit_wins(self) -> None:
        assert resolve_config_value("/explicit", "TEST_ENV", "/default") == "/explicit"

    def test_env_var_used(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_CFG_VAR", "/from_env")
        assert resolve_config_value(None, "TEST_CFG_VAR", "/default") == "/from_env"

    def test_default_fallback(self) -> None:
        assert resolve_config_value(None, "UNLIKELY_ENV_VAR_XYZ", "/default") == "/default"


class TestConfigEntry:
    def test_resolve_explicit(self) -> None:
        entry = ConfigEntry(name="store", env_var="TEST_STORE", default="/def")
        assert entry.resolve("/override") == "/override"

    def test_resolve_default(self) -> None:
        entry = ConfigEntry(name="store", env_var="UNLIKELY_ENV_XYZ", default="/def")
        assert entry.resolve() == "/def"
