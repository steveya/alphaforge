#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PIT_CODE_PREFIXES = (
    "alphaforge/pit/",
    "alphaforge/data/pit_source.py",
    "alphaforge/__init__.py",
)

PIT_DOC_PREFIXES = (
    "docs/guides/pit",
    "docs/api/pit-",
    "mkdocs.yml",
)

PIT_CONTRACT_AFFECTING_PREFIXES = (
    "alphaforge/pit/accessor.py",
    "alphaforge/pit/validation.py",
    "alphaforge/pit/contract.py",
    "alphaforge/pit/models.py",
)

PIT_CONTRACT_DOC_FILES = {
    "docs/guides/pit-api-contract.md",
    "docs/guides/pit-migrations.md",
}

CHANGELOG_FILES = {"CHANGELOG.md", "docs/changelog.md"}


def _normalize_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for item in paths:
        value = item.strip()
        if not value:
            continue
        out.append(str(Path(value).as_posix()))
    return out


def main(argv: list[str]) -> int:
    changed_files = _normalize_paths(argv[1:])
    if not changed_files:
        print("No changed files detected. PIT docs sync guard skipped.")
        return 0

    pit_code_changed = any(
        any(path.startswith(prefix) for prefix in PIT_CODE_PREFIXES) for path in changed_files
    )
    if not pit_code_changed:
        print("No PIT public-surface code changes detected.")
        return 0

    pit_docs_changed = any(
        any(path.startswith(prefix) for prefix in PIT_DOC_PREFIXES) for path in changed_files
    )
    changelog_changed = any(path in CHANGELOG_FILES for path in changed_files)
    pit_contract_code_changed = any(
        any(path.startswith(prefix) for prefix in PIT_CONTRACT_AFFECTING_PREFIXES)
        for path in changed_files
    )
    pit_contract_docs_changed = all(path in changed_files for path in PIT_CONTRACT_DOC_FILES)

    if pit_docs_changed and changelog_changed and (
        not pit_contract_code_changed or pit_contract_docs_changed
    ):
        print("PIT docs sync guard passed.")
        return 0

    missing: list[str] = []
    if not pit_docs_changed:
        missing.append("PIT guide/API docs updates")
    if not changelog_changed:
        missing.append("CHANGELOG updates")
    if pit_contract_code_changed and not pit_contract_docs_changed:
        missing.append("PIT contract+migration docs updates")

    print(
        "PIT docs sync guard failed: public PIT code changed without required docs/changelog "
        f"updates ({', '.join(missing)})."
    )
    print("Changed files:")
    for path in changed_files:
        print(f"- {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
