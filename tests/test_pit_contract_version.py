from pathlib import Path

from alphaforge import PIT_CONTRACT_VERSION, get_pit_contract_version
from alphaforge.pit.contract import PIT_CONTRACT_VERSION as PIT_CONTRACT_VERSION_INNER


def test_contract_version_api_stable():
    assert isinstance(PIT_CONTRACT_VERSION, str)
    assert PIT_CONTRACT_VERSION == PIT_CONTRACT_VERSION_INNER
    assert get_pit_contract_version() == PIT_CONTRACT_VERSION


def test_pit_migration_guide_exists_and_mentions_current_version():
    path = Path(__file__).resolve().parents[1] / "docs" / "guides" / "pit-migrations.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert PIT_CONTRACT_VERSION in text
