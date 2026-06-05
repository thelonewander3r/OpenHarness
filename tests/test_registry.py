"""Tests for the hot-reloading frontier model registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from decomphose.config import HarnessSettings, resolve_registry_path
from decomphose.registry import ModelRegistry
from decomphose.utils.errors import HarnessError

VALID_CONFIG = {
    "version": 1,
    "updatedAt": "2026-06-05",
    "models": [
        {
            "id": "test/model-a",
            "provider": "openrouter",
            "capabilityRank": 90,
            "contextWindow": 100000,
        }
    ],
}


def write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def bump_mtime(path: Path) -> None:
    """Force a visible mtime change regardless of filesystem timestamp resolution."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_loads_valid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    config = registry.load()
    assert config.models[0].id == "test/model-a"


def test_caches_when_unchanged(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    assert registry.load() is registry.load()


def test_hot_reloads_on_change(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    assert registry.load().models[0].id == "test/model-a"

    updated = {
        **VALID_CONFIG,
        "version": 2,
        "models": [{**VALID_CONFIG["models"][0], "id": "test/model-b"}],
    }
    write_config(config_file, updated)
    bump_mtime(config_file)

    reloaded = registry.load()
    assert reloaded.version == 2
    assert reloaded.models[0].id == "test/model-b"


def test_invalid_edit_serves_last_good(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    good = registry.load()

    config_file.write_text("{not valid json", encoding="utf-8")
    bump_mtime(config_file)
    assert registry.load() is good

    # Schema violation is also tolerated once a good config exists.
    write_config(config_file, {"version": 3, "updatedAt": "x", "models": "oops"})
    bump_mtime(config_file)
    assert registry.load() is good


def test_empty_models_serves_last_good(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    good = registry.load()

    write_config(config_file, {**VALID_CONFIG, "models": []})
    bump_mtime(config_file)
    assert registry.load() is good


def test_recovers_after_invalid_edit(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    registry.load()

    config_file.write_text("{broken", encoding="utf-8")
    bump_mtime(config_file)
    registry.load()

    fixed = {**VALID_CONFIG, "version": 5}
    write_config(config_file, fixed)
    bump_mtime(config_file)
    assert registry.load().version == 5


def test_invalid_initial_load_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    config_file.write_text("{broken", encoding="utf-8")

    registry = ModelRegistry(config_file)
    with pytest.raises(HarnessError) as exc_info:
        registry.load()
    assert exc_info.value.code == "INVALID_REGISTRY"


def test_missing_file_initial_load_raises(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "missing.json")
    with pytest.raises(HarnessError):
        registry.load()


def test_missing_file_after_good_load_serves_last_good(tmp_path: Path) -> None:
    config_file = tmp_path / "models.json"
    write_config(config_file, VALID_CONFIG)

    registry = ModelRegistry(config_file)
    good = registry.load()

    config_file.unlink()
    assert registry.load() is good


def test_registry_path_pluggable_via_settings(tmp_path: Path) -> None:
    custom = tmp_path / "custom-models.json"
    settings = HarnessSettings(harness_frontier_models_path=str(custom))
    assert resolve_registry_path(settings) == custom.resolve()


def test_registry_path_defaults_to_bundled_config() -> None:
    settings = HarnessSettings(harness_frontier_models_path="")
    assert resolve_registry_path(settings).name == "frontier-models.json"
