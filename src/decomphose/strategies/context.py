from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decomphose.clients.openrouter import OpenRouterClient
from decomphose.config import HarnessSettings
from decomphose.types import HarnessRequestMeta


@dataclass
class StrategyContext:
    settings: HarnessSettings
    client: OpenRouterClient
    meta: HarnessRequestMeta
    body: dict[str, Any]
