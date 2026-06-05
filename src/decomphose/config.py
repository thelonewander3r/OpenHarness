from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from decomphose.registry import ModelRegistry
from decomphose.types import FrontierModelsConfig


def _resolve_project_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "config" / "frontier-models.json").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "config" / "frontier-models.json").exists():
        return cwd
    return candidate


FRONTIER_CONFIG_PATH = _resolve_project_root() / "config" / "frontier-models.json"


class HarnessSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    harness_host: str = "0.0.0.0"
    harness_port: int = 3100
    harness_decomp_model: str = "anthropic/claude-3.5-haiku"
    harness_worker_model: str = "deepseek/deepseek-chat"
    harness_auditor_model: str = "anthropic/claude-3.5-haiku"
    harness_max_auditor_retries: int = 3
    harness_frontier_models_path: str = ""
    harness_otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""


@lru_cache
def get_settings() -> HarnessSettings:
    settings = HarnessSettings()
    if not settings.openrouter_api_key:
        print(
            "[decomphose] OPENROUTER_API_KEY is unset — upstream calls will fail until configured."
        )
    return settings


def resolve_registry_path(settings: HarnessSettings) -> Path:
    if settings.harness_frontier_models_path:
        return Path(settings.harness_frontier_models_path).expanduser().resolve()
    return FRONTIER_CONFIG_PATH


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(resolve_registry_path(get_settings()))


def load_frontier_models() -> FrontierModelsConfig:
    """Current frontier model registry — hot-reloads when the backing file changes."""
    return get_model_registry().load()


def clear_frontier_cache() -> None:
    get_model_registry.cache_clear()
