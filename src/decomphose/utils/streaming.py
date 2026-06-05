from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

SSE_DONE = "data: [DONE]\n\n"

# Character size of each synthetic content delta when streaming a compiled response.
SYNTHETIC_CHUNK_CHARS = 512


def sse_event(payload: str) -> str:
    return f"data: {payload}\n\n"


async def sse_passthrough(chunks: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Re-encode upstream ChatCompletionChunk objects as an SSE stream.

    Mid-stream upstream failures are surfaced as a final SSE error event —
    headers are already sent at that point, so the status code cannot change.
    """
    try:
        async for chunk in chunks:
            yield sse_event(chunk.model_dump_json())
    except Exception as exc:  # noqa: BLE001 — stream is already open; report inline.
        yield sse_event(
            json.dumps(
                {"error": {"message": str(exc), "type": "UPSTREAM_STREAM_ERROR"}}
            )
        )
    finally:
        yield SSE_DONE


async def sse_synthetic_completion(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> AsyncIterator[str]:
    """Emit an already-compiled response as an OpenAI-shaped chat.completion.chunk stream."""

    def chunk_body(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )

    yield sse_event(chunk_body({"role": "assistant"}))
    for offset in range(0, len(content), SYNTHETIC_CHUNK_CHARS):
        yield sse_event(chunk_body({"content": content[offset : offset + SYNTHETIC_CHUNK_CHARS]}))
    yield sse_event(chunk_body({}, finish_reason="stop"))
    yield SSE_DONE
