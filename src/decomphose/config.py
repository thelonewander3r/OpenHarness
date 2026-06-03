from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

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


@lru_cache
def get_settings() -> HarnessSettings:
    settings = HarnessSettings()
    if not settings.openrouter_api_key:
        print(
            "[decomphose] OPENROUTER_API_KEY is unset — upstream calls will fail until configured."
        )
    return settings


@lru_cache
def load_frontier_models() -> FrontierModelsConfig:
    raw = FRONTIER_CONFIG_PATH.read_text(encoding="utf-8")
    return FrontierModelsConfig.model_validate(json.loads(raw))


def clear_frontier_cache() -> None:
    load_frontier_models.cache_clear()
