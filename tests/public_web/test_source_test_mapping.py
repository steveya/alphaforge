from __future__ import annotations

from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2] / "alphaforge" / "data" / "public_web"
TEST_DIR = Path(__file__).resolve().parent

NON_SOURCE_MODULES = {
    "__init__",
    "archive",
    "base",
    "finalize",
    "http",
    "parsing",
    "registry_api",
    "utils",
    "registry",
    "registry_loader",
    "schema_helpers",
    "tabular",
}

NON_MAPPING_TEST_FILES = {
    "__init__.py",
    "_fake_http.py",
    "test_live_sources.py",
    "test_public_web_foundation.py",
    "test_source_test_mapping.py",
}


def _source_modules() -> set[str]:
    modules = {
        path.stem
        for path in SOURCE_DIR.glob("*.py")
        if path.stem not in NON_SOURCE_MODULES and not path.stem.startswith("_")
    }
    return modules


def _mapped_test_modules() -> set[str]:
    mapped = {
        path.stem.removeprefix("test_")
        for path in TEST_DIR.glob("test_*.py")
        if path.name not in NON_MAPPING_TEST_FILES
    }
    return mapped


def test_every_source_module_has_matching_test_file() -> None:
    source_modules = _source_modules()
    mapped_tests = _mapped_test_modules()

    missing_test_files = sorted(source_modules - mapped_tests)

    assert not missing_test_files, (
        "Missing test files for source modules in alphaforge.data.public_web: "
        + ", ".join(f"test_{name}.py" for name in missing_test_files)
    )


def test_every_mapping_test_file_has_matching_source_module() -> None:
    source_modules = _source_modules()
    mapped_tests = _mapped_test_modules()

    orphaned_test_files = sorted(mapped_tests - source_modules)

    assert not orphaned_test_files, (
        "Orphaned test files with no matching source module in alphaforge.data.public_web: "
        + ", ".join(f"test_{name}.py" for name in orphaned_test_files)
    )
