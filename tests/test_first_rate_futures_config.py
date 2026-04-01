from __future__ import annotations

from pathlib import Path

import pytest

from alphaforge.futures import FirstRateFuturesConfig


def test_futures_config_explicit_overrides_yaml_and_env(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "futures.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"source_dir: {tmp_path / 'yaml_source'}",
                f"artifact_root: {tmp_path / 'yaml_artifacts'}",
                "roll_min_consecutive_sessions: 3",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHAFORGE_FRD_FUTURES_SOURCE_DIR", str(tmp_path / "env_source"))
    monkeypatch.setenv("ALPHAFORGE_FRD_FUTURES_ARTIFACT_ROOT", str(tmp_path / "env_artifacts"))

    cfg = FirstRateFuturesConfig.resolve(
        config_path=yaml_path,
        source_dir=tmp_path / "explicit_source",
        artifact_root=tmp_path / "explicit_artifacts",
        roll_min_consecutive_sessions=4,
    )

    assert cfg.source_dir == (tmp_path / "explicit_source").resolve()
    assert cfg.artifact_root == (tmp_path / "explicit_artifacts").resolve()
    assert cfg.roll_min_consecutive_sessions == 4


def test_futures_config_uses_yaml_before_env(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "futures.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"source_dir: {tmp_path / 'yaml_source'}",
                f"artifact_root: {tmp_path / 'yaml_artifacts'}",
                "bar_minutes: 5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHAFORGE_FRD_FUTURES_SOURCE_DIR", str(tmp_path / "env_source"))
    monkeypatch.setenv("ALPHAFORGE_FRD_FUTURES_ARTIFACT_ROOT", str(tmp_path / "env_artifacts"))

    cfg = FirstRateFuturesConfig.from_yaml(yaml_path)

    assert cfg.source_dir == (tmp_path / "yaml_source").resolve()
    assert cfg.artifact_root == (tmp_path / "yaml_artifacts").resolve()


def test_futures_config_uses_env_config_path(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "futures.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"source_dir: {tmp_path / 'yaml_source'}",
                f"artifact_root: {tmp_path / 'yaml_artifacts'}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHAFORGE_FRD_FUTURES_CONFIG", str(yaml_path))

    cfg = FirstRateFuturesConfig.from_env()

    assert cfg.config_path == yaml_path.resolve()
    assert cfg.source_dir == (tmp_path / "yaml_source").resolve()


def test_futures_config_requires_source_and_artifact_paths(monkeypatch) -> None:
    monkeypatch.delenv("ALPHAFORGE_FRD_FUTURES_SOURCE_DIR", raising=False)
    monkeypatch.delenv("ALPHAFORGE_FRD_FUTURES_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("ALPHAFORGE_FRD_FUTURES_CONFIG", raising=False)

    with pytest.raises(ValueError, match="Missing required futures configuration values"):
        FirstRateFuturesConfig.from_env()


def test_public_examples_do_not_embed_local_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (
        repo_root / ".env.example",
        repo_root / "examples" / "first_rate_futures.yaml",
        repo_root / "docs" / "guides" / "first-rate-futures.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "Projects/fut_contract_price_5m" not in text
