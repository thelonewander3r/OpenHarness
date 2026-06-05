"""Smoke tests for the async client wiring and SSE streaming paths.

Uses a fake upstream client injected into create_app — no network, no API key.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from decomphose.server import create_app

COMPLETIONS_URL = "/v1/chat/completions"

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Summarize the design.\n\n[context:brief]\nA short brief."},
    ],
}


class FakeChunk:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump_json(self) -> str:
        return json.dumps(self._payload)


class FakeCompletion:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeUpstreamClient:
    """Mimics OpenRouterClient's async surface without network access."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        system = messages[0]["content"]
        if "decomposition engine" in system:
            return json.dumps(
                {
                    "microTasks": [
                        {
                            "id": "task-1",
                            "title": "Step one",
                            "instruction": "Do step one",
                            "relevantContextKeys": ["brief"],
                        },
                        {
                            "id": "task-2",
                            "title": "Step two",
                            "instruction": "Do step two",
                            "relevantContextKeys": ["brief"],
                        },
                    ]
                }
            )
        if "Goal Auditor" in system:
            return "PASS"
        return "worker output"

    async def forward_raw(self, body: dict[str, Any]) -> FakeCompletion:
        return FakeCompletion(
            {
                "id": "cmpl-1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def stream_raw(self, body: dict[str, Any]):
        for i in range(3):
            yield FakeChunk(
                {
                    "id": "cmpl-1",
                    "object": "chat.completion.chunk",
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "delta": {"content": f"part-{i} "}, "finish_reason": None}
                    ],
                }
            )

    async def close(self) -> None:
        pass


def make_client() -> TestClient:
    return TestClient(create_app(client=FakeUpstreamClient()))


def collect_sse_data(lines: list[str]) -> list[str]:
    return [line[len("data: ") :] for line in lines if line.startswith("data: ")]


def test_accuracy_non_stream() -> None:
    with make_client() as client:
        res = client.post(
            COMPLETIONS_URL,
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "accuracy"},
        )
    assert res.status_code == 200
    assert res.headers["x-harness-strategy"] == "accuracy"
    assert res.headers["x-harness-selected-model"]
    assert res.json()["choices"][0]["message"]["content"] == "hello"


def test_accuracy_stream_passthrough() -> None:
    with make_client() as client, client.stream(
        "POST",
        COMPLETIONS_URL,
        json={**REQUEST_BODY, "stream": True},
        headers={"X-Harness-Strategy": "accuracy"},
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = collect_sse_data(list(res.iter_lines()))

    assert events[-1] == "[DONE]"
    deltas = [
        json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1]
    ]
    assert "".join(deltas) == "part-0 part-1 part-2 "


def test_affordability_non_stream() -> None:
    with make_client() as client:
        res = client.post(
            COMPLETIONS_URL,
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )
    assert res.status_code == 200
    assert res.headers["x-harness-decomposition-steps"] == "2"
    content = res.json()["choices"][0]["message"]["content"]
    assert "## Step one" in content
    assert "worker output" in content


def test_affordability_stream_synthetic() -> None:
    with make_client() as client, client.stream(
        "POST",
        COMPLETIONS_URL,
        json={**REQUEST_BODY, "stream": True},
        headers={"X-Harness-Strategy": "affordability"},
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        assert res.headers["x-harness-decomposition-steps"] == "2"
        events = collect_sse_data(list(res.iter_lines()))

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "## Step one" in content
    assert "## Step two" in content


def test_missing_strategy_header_rejected() -> None:
    with make_client() as client:
        res = client.post(COMPLETIONS_URL, json=REQUEST_BODY)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MISSING_STRATEGY"
