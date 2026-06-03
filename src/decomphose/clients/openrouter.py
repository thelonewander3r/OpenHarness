from __future__ import annotations

from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from decomphose.config import HarnessSettings
from decomphose.utils.errors import StrategyError


class OpenRouterClient:
    def __init__(self, settings: HarnessSettings) -> None:
        self._settings = settings
        # Placeholder allows the proxy to boot for /health; upstream calls need a real key.
        api_key = settings.openrouter_api_key or "missing-configured-key"
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/decomphose/harness",
                "X-Title": "Decomphose Harness",
            },
        )

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise StrategyError("Upstream returned empty completion", "EMPTY_COMPLETION")
            return content
        except StrategyError:
            raise
        except Exception as exc:
            raise StrategyError("OpenRouter completion failed", "UPSTREAM_ERROR", cause=exc) from exc

    def forward_raw(self, body: dict[str, Any]) -> ChatCompletion:
        try:
            return self._client.chat.completions.create(**body)
        except Exception as exc:
            raise StrategyError("OpenRouter forward failed", "UPSTREAM_ERROR", cause=exc) from exc
