"""Tests for structured decomposition validation (Pydantic) and its retry/fallback path."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from decomphose.server import create_app
from decomphose.utils.decomposition import (
    MAX_MICRO_TASKS,
    parse_decomposition_plan,
    strip_json_fences,
)
from decomphose.utils.errors import DecompositionError

VALID_PLAN = {
    "microTasks": [
        {
            "id": "task-1",
            "title": "Step one",
            "instruction": "Do step one",
            "relevantContextKeys": ["brief"],
        },
        {"instruction": "Do step two"},
    ]
}


# --- parse_decomposition_plan unit tests ---


def test_parse_valid_plan_normalizes_defaults() -> None:
    tasks = parse_decomposition_plan(json.dumps(VALID_PLAN))
    assert len(tasks) == 2
    assert tasks[0].id == "task-1"
    assert tasks[0].title == "Step one"
    assert tasks[0].relevant_context_keys == ["brief"]
    # Missing id/title filled from index.
    assert tasks[1].id == "task-2"
    assert tasks[1].title == "Step 2"
    assert tasks[1].index == 1
    assert tasks[1].relevant_context_keys == []


def test_parse_strips_markdown_fences() -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    assert len(parse_decomposition_plan(fenced)) == 2


def test_strip_json_fences_passthrough() -> None:
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(DecompositionError) as exc_info:
        parse_decomposition_plan("not json at all {")
    assert exc_info.value.code == "INVALID_DECOMPOSITION"


def test_parse_missing_instruction_raises() -> None:
    bad = {"microTasks": [{"id": "task-1", "title": "No instruction"}]}
    with pytest.raises(DecompositionError):
        parse_decomposition_plan(json.dumps(bad))


def test_parse_blank_instruction_raises() -> None:
    bad = {"microTasks": [{"instruction": "   "}]}
    with pytest.raises(DecompositionError):
        parse_decomposition_plan(json.dumps(bad))


def test_parse_wrong_shape_raises() -> None:
    with pytest.raises(DecompositionError):
        parse_decomposition_plan(json.dumps({"microTasks": "oops"}))


def test_parse_empty_plan_returns_empty_list() -> None:
    assert parse_decomposition_plan(json.dumps({"microTasks": []})) == []


def test_parse_truncates_over_decomposition() -> None:
    big = {"microTasks": [{"instruction": f"Step {i}"} for i in range(20)]}
    tasks = parse_decomposition_plan(json.dumps(big))
    assert len(tasks) == MAX_MICRO_TASKS


# --- strategy-level retry / fallback tests ---

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [
        {"role": "user", "content": "Do the master task.\n\n[context:brief]\nA short brief."},
    ],
}


class ScriptedFakeClient:
    """Fake upstream whose decomposition responses come from a fixed script."""

    def __init__(self, decomposition_responses: list[str]) -> None:
        self._decomposition_responses = list(decomposition_responses)
        self.decomposition_calls: list[list[dict[str, Any]]] = []

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
            self.decomposition_calls.append(messages)
            return self._decomposition_responses.pop(0)
        if "Goal Auditor" in system:
            return "PASS"
        return "worker output"

    async def close(self) -> None:
        pass


def post_affordability(fake: ScriptedFakeClient):
    with TestClient(create_app(client=fake)) as client:
        return client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )


def test_malformed_then_valid_decomposition_retries() -> None:
    fake = ScriptedFakeClient(["```oops not json", json.dumps(VALID_PLAN)])
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-decomposition-steps"] == "2"
    assert len(fake.decomposition_calls) == 2
    # Retry must include the bad output and corrective feedback.
    retry_messages = fake.decomposition_calls[1]
    assert retry_messages[-2]["role"] == "assistant"
    assert "invalid" in retry_messages[-1]["content"]


def test_persistently_malformed_decomposition_falls_back() -> None:
    fake = ScriptedFakeClient(["{bad", "{still bad"])
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-decomposition-steps"] == "1"
    content = res.json()["choices"][0]["message"]["content"]
    assert "## Execute master task" in content
    assert "worker output" in content


def test_valid_but_empty_plan_falls_back_without_retry() -> None:
    fake = ScriptedFakeClient([json.dumps({"microTasks": []})])
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-decomposition-steps"] == "1"
    assert len(fake.decomposition_calls) == 1
