"""Tests for per-request cost accounting (usage ledger, pricing, headers)."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from decomphose.accounting import UsageLedger, current_ledger
from decomphose.server import create_app
from decomphose.types import ModelPricingConfig

PRICING = ModelPricingConfig.model_validate(
    {
        "version": 1,
        "updatedAt": "2026-06-05",
        "models": [
            {"id": "cheap/model", "promptPerMtok": 1.0, "completionPerMtok": 2.0},
            {"id": "big/model", "promptPerMtok": 10.0, "completionPerMtok": 30.0},
        ],
    }
)


# --- UsageLedger unit tests ---


def test_ledger_computes_cost_from_pricing() -> None:
    ledger = UsageLedger(pricing=PRICING)
    ledger.record("cheap/model", 1_000_000, 500_000)

    assert ledger.prompt_tokens == 1_000_000
    assert ledger.completion_tokens == 500_000
    assert ledger.cost_usd == 1.0 + 1.0  # 1M prompt @ $1 + 0.5M completion @ $2
    assert ledger.coverage == "full"


def test_ledger_unknown_model_is_partial_coverage() -> None:
    ledger = UsageLedger(pricing=PRICING)
    ledger.record("cheap/model", 100, 100)
    ledger.record("mystery/model", 100, 100)

    assert ledger.coverage == "partial"
    assert ledger.llm_calls == 2
    # Known cost still summed; unknown contributes nothing rather than guessing.
    assert ledger.cost_usd > 0


def test_ledger_without_pricing_has_no_cost() -> None:
    ledger = UsageLedger(pricing=None)
    ledger.record("any/model", 100, 100)

    assert ledger.coverage == "none"
    headers = ledger.headers()
    assert headers["x-harness-cost-coverage"] == "none"
    assert "x-harness-cost-usd" not in headers
    assert headers["x-harness-prompt-tokens"] == "100"


def test_empty_ledger_emits_no_headers() -> None:
    assert UsageLedger(pricing=PRICING).headers() == {}


# --- end-to-end through the server ---

PLAN = json.dumps(
    {
        "microTasks": [
            {"id": "task-1", "title": "Step one", "instruction": "Do step one"},
            {"id": "task-2", "title": "Step two", "instruction": "Do step two"},
        ]
    }
)

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Do the master task."}],
}


class MeteredFakeClient:
    """Records fixed usage into the request ledger, like the real client does."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        ledger = current_ledger()
        assert ledger is not None, "server must start a ledger before upstream calls"
        ledger.record(model, 100, 50)

        system = messages[0]["content"]
        if "decomposition engine" in system:
            return PLAN
        if "Goal Auditor" in system:
            return "PASS"
        return "worker output"

    async def close(self) -> None:
        pass


def test_affordability_reports_cost_headers_and_usage() -> None:
    with TestClient(create_app(client=MeteredFakeClient())) as client:
        res = client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )

    assert res.status_code == 200
    # 1 decomposition + 2 workers + 2 auditors = 5 upstream calls.
    assert res.headers["x-harness-llm-calls"] == "5"
    assert res.headers["x-harness-prompt-tokens"] == "500"
    assert res.headers["x-harness-completion-tokens"] == "250"
    # Repo pricing file covers the default decomp/auditor/worker models.
    assert res.headers["x-harness-cost-coverage"] == "full"
    assert float(res.headers["x-harness-cost-usd"]) > 0

    usage = res.json()["usage"]
    assert usage["prompt_tokens"] == 500
    assert usage["completion_tokens"] == 250
    assert usage["total_tokens"] == 750


def test_ledger_is_isolated_per_request() -> None:
    with TestClient(create_app(client=MeteredFakeClient())) as client:
        first = client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )
        second = client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )

    # Totals must not accumulate across requests.
    assert first.headers["x-harness-llm-calls"] == second.headers["x-harness-llm-calls"]
    assert first.headers["x-harness-prompt-tokens"] == second.headers["x-harness-prompt-tokens"]
