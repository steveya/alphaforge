"""Tests for the generic EntityRegistry."""
from __future__ import annotations

import pytest

from alphaforge.registry import EntityEntry, EntityRegistry


class TestEntityRegistry:
    def test_register_and_get(self) -> None:
        reg = EntityRegistry()
        entry = EntityEntry(entity_id="eur", asset_class="fx", source_keys={"cot": "k1"})
        reg.register("eur", entry)
        assert reg.get("eur") is entry

    def test_get_unknown_raises(self) -> None:
        reg = EntityRegistry()
        with pytest.raises(KeyError, match="Unknown entity"):
            reg.get("missing")

    def test_entities_filter_by_asset_class(self) -> None:
        reg = EntityRegistry()
        reg.register("eur", EntityEntry("eur", "fx"))
        reg.register("usd_irs_5y", EntityEntry("usd_irs_5y", "rates"))
        assert reg.entities("fx") == ["eur"]
        assert reg.entities("rates") == ["usd_irs_5y"]
        assert sorted(reg.entities()) == ["eur", "usd_irs_5y"]

    def test_source_key_and_all_source_keys(self) -> None:
        reg = EntityRegistry()
        reg.register("eur", EntityEntry("eur", "fx", source_keys={"cot": "cot_eur", "dtcc": "dtcc_eur"}))
        reg.register("gbp", EntityEntry("gbp", "fx", source_keys={"cot": "cot_gbp"}))
        assert reg.source_key("eur", "cot") == "cot_eur"
        assert reg.all_source_keys("cot") == {"eur": "cot_eur", "gbp": "cot_gbp"}
        assert reg.all_source_keys("dtcc") == {"eur": "dtcc_eur"}
