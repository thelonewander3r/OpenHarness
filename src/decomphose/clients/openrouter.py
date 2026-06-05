from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from decomphose.config import HarnessSettings
from decomphose.telemetry import get_tracer
from decomphose.utils.errors import StrategyError


class OpenRouterClient:
    def __init__(self, settings: HarnessSettings) -> None:
        self._settings = settings
        # Placeholder allows the proxy to boot for /health; upstream calls need a real key.
        api_key = settings.openrouter_api_key or "missing-configured-key"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/decomphose/harness",
                "X-Title": "Decomphose Harness",
            },
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        with get_tracer().start_as_current_span("upstream.complete") as span:
            span.set_attribute("llm.model", model)
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise StrategyError("Upstream returned empty completion", "EMPTY_COMPLETION")
                span.set_attribute("llm.response_chars", len(content))
                return content
            except StrategyError:
                raise
            except Exception as exc:
                raise StrategyError(
                    "OpenRouter completion failed", "UPSTREAM_ERROR", cause=exc
                ) from exc

    async def forward_raw(self, body: dict[str, Any]) -> ChatCompletion:
        try:
            # Non-streaming forward — streaming requests go through stream_raw.
            return await self._client.chat.completions.create(**{**body, "stream": False})
        except Exception as exc:
            raise StrategyError("OpenRouter forward failed", "UPSTREAM_ERROR", cause=exc) from exc

    async def stream_raw(self, body: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        try:
            stream = await self._client.chat.completions.create(**{**body, "stream": True})
        except Exception as exc:
            raise StrategyError("OpenRouter stream failed", "UPSTREAM_ERROR", cause=exc) from exc
        async for chunk in stream:
            yield chunk

    async def close(self) -> None:
        await self._client.close()
