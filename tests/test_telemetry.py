"""Tests for OpenTelemetry spans emitted per request and per micro-task.

Installs a real TracerProvider with an in-memory exporter at import time;
other test modules run with the tracer regardless, which is harmless —
each test clears the exporter before acting.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from decomphose.server import create_app

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [
        {"role": "user", "content": "Do the task.\n\n[context:brief]\nA short brief."},
    ],
}

PLAN = json.dumps(
    {
        "microTasks": [
            {"id": "task-1", "title": "Step one", "instruction": "Do step one"},
            {"id": "task-2", "title": "Step two", "instruction": "Do step two"},
        ]
    }
)


class FakeUpstreamClient:
    def __init__(self, auditor_first_verdict: str = "PASS") -> None:
        self._auditor_verdicts = [auditor_first_verdict]

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
            return PLAN
        if "Goal Auditor" in system:
            return self._auditor_verdicts.pop(0) if self._auditor_verdicts else "PASS"
        return "worker output"

    async def forward_raw(self, body: dict[str, Any]):
        class FakeCompletion:
            @staticmethod
            def model_dump() -> dict[str, Any]:
                return {
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

        return FakeCompletion()

    async def close(self) -> None:
        pass


def run_request(strategy: str, fake: FakeUpstreamClient | None = None) -> None:
    EXPORTER.clear()
    with TestClient(create_app(client=fake or FakeUpstreamClient())) as client:
        res = client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": strategy},
        )
    assert res.status_code == 200


def spans_by_name() -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for span in EXPORTER.get_finished_spans():
        grouped.setdefault(span.name, []).append(span)
    return grouped


def test_affordability_emits_pipeline_spans() -> None:
    run_request("affordability")
    spans = spans_by_name()

    assert len(spans["harness.request"]) == 1
    request_span = spans["harness.request"][0]
    assert request_span.attributes["harness.strategy"] == "affordability"

    decompose = spans["affordability.decompose"][0]
    assert decompose.attributes["harness.micro_task_count"] == 2
    assert decompose.parent.span_id == request_span.context.span_id

    micro_tasks = spans["affordability.micro_task"]
    assert [s.attributes["harness.task.id"] for s in micro_tasks] == ["task-1", "task-2"]
    assert all(s.parent.span_id == request_span.context.span_id for s in micro_tasks)
    assert all(s.attributes["harness.auditor_retries"] == 0 for s in micro_tasks)


def test_auditor_reject_recorded_as_span_event() -> None:
    run_request(
        "affordability",
        FakeUpstreamClient(auditor_first_verdict="REJECT: missing detail"),
    )
    spans = spans_by_name()

    task_one = spans["affordability.micro_task"][0]
    event_names = [e.name for e in task_one.events]
    assert "auditor.reject" in event_names
    assert "auditor.pass" in event_names
    assert task_one.attributes["harness.auditor_retries"] == 1


def test_accuracy_records_routing_attributes() -> None:
    run_request("accuracy")
    spans = spans_by_name()

    request_span = spans["harness.request"][0]
    assert request_span.attributes["harness.strategy"] == "accuracy"
    assert request_span.attributes["harness.selected_model"]
    assert "harness.estimated_tokens" in request_span.attributes
