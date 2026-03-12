from __future__ import annotations

PITContractVersion = str

PIT_CONTRACT_VERSION: PITContractVersion = "2.1.0"


def get_pit_contract_version() -> PITContractVersion:
    return PIT_CONTRACT_VERSION
