"""Tests for Factory Router-style per-micro-task model routing and escalation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from decomphose.config import clear_frontier_cache, get_settings
from decomphose.router import build_worker_escalation_path
from decomphose.server import create_app
from decomphose.types import WorkerModelsConfig

WORKER_CONFIG = {
    "version": 1,
    "updatedAt": "2026-06-05",
    "tiers": ["routine", "standard", "complex"],
    "models": [
        {"id": "cheap/routine-a", "tier": "routine", "costRank": 1},
        {"id": "cheap/routine-b", "tier": "routine", "costRank": 2},
        {"id": "mid/standard-a", "tier": "standard", "costRank": 3},
        {"id": "big/complex-a", "tier": "complex", "costRank": 9},
    ],
}

CONFIG = WorkerModelsConfig.model_validate(WORKER_CONFIG)


# --- build_worker_escalation_path unit tests ---


def test_routing_disabled_without_config() -> None:
    assert build_worker_escalation_path(None, "routine", "default/model") == ["default/model"]


def test_routine_task_gets_full_escalation_path() -> None:
    path = build_worker_escalation_path(CONFIG, "routine", "default/model")
    assert path == ["cheap/routine-a", "mid/standard-a", "big/complex-a"]


def test_complex_task_starts_at_top_tier() -> None:
    assert build_worker_escalation_path(CONFIG, "complex", "default/model") == ["big/complex-a"]


def test_unknown_complexity_falls_back_to_standard_tier() -> None:
    path = build_worker_escalation_path(CONFIG, "galactic", "default/model")
    assert path == ["mid/standard-a", "big/complex-a"]


def test_cheapest_model_wins_within_tier() -> None:
    path = build_worker_escalation_path(CONFIG, "routine", "default/model")
    assert path[0] == "cheap/routine-a"  # costRank 1 beats costRank 2


def test_empty_tier_is_skipped() -> None:
    config = WorkerModelsConfig.model_validate(
        {
            **WORKER_CONFIG,
            "models": [m for m in WORKER_CONFIG["models"] if m["tier"] != "standard"],
        }
    )
    path = build_worker_escalation_path(config, "routine", "default/model")
    assert path == ["cheap/routine-a", "big/complex-a"]


# --- end-to-end routing through the affordability strategy ---

PLAN = json.dumps(
    {
        "microTasks": [
            {
                "id": "task-1",
                "title": "Boilerplate",
                "instruction": "Do mechanical step",
                "complexity": "routine",
            },
            {
                "id": "task-2",
                "title": "Architecture",
                "instruction": "Design the system",
                "complexity": "complex",
            },
        ]
    }
)

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Do the master task."}],
}


class RoutingFakeClient:
    """Records which worker model each micro-task attempt used."""

    def __init__(self, reject_first_for_task: str | None = None) -> None:
        self.worker_calls: list[tuple[str, str]] = []  # (model, task title line)
        self._reject_first_for_task = reject_first_for_task
        self._rejected = False

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
            user = messages[1]["content"]
            if (
                self._reject_first_for_task
                and not self._rejected
                and self._reject_first_for_task in user
            ):
                self._rejected = True
                return "REJECT: not good enough"
            return "PASS"
        self.worker_calls.append((model, messages[1]["content"].splitlines()[0]))
        return "worker output"

    async def close(self) -> None:
        pass


@pytest.fixture()
def routed_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App wired to a temp worker-models registry via HARNESS_WORKER_MODELS_PATH."""
    config_file = tmp_path / "worker-models.json"
    config_file.write_text(json.dumps(WORKER_CONFIG), encoding="utf-8")
    monkeypatch.setenv("HARNESS_WORKER_MODELS_PATH", str(config_file))
    get_settings.cache_clear()
    clear_frontier_cache()
    yield
    get_settings.cache_clear()
    clear_frontier_cache()


def post_affordability(fake: RoutingFakeClient):
    with TestClient(create_app(client=fake)) as client:
        return client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )


def test_tasks_route_to_their_tier(routed_app, monkeypatch) -> None:
    fake = RoutingFakeClient()
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-router"] == "enabled"
    assert res.headers["x-harness-router-escalations"] == "0"
    assert res.headers["x-harness-worker-models"] == "cheap/routine-a,big/complex-a"
    models = [model for model, _ in fake.worker_calls]
    assert models == ["cheap/routine-a", "big/complex-a"]


def test_auditor_reject_escalates_one_tier(routed_app) -> None:
    fake = RoutingFakeClient(reject_first_for_task="Do mechanical step")
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-router-escalations"] == "1"
    models = [model for model, _ in fake.worker_calls]
    # task-1: routine attempt rejected -> escalated to standard tier; task-2: complex.
    assert models == ["cheap/routine-a", "mid/standard-a", "big/complex-a"]
    assert "cheap/routine-a,mid/standard-a,big/complex-a" == res.headers["x-harness-worker-models"]


def test_router_disabled_uses_static_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_WORKER_MODELS_PATH", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    clear_frontier_cache()
    try:
        fake = RoutingFakeClient()
        res = post_affordability(fake)

        assert res.status_code == 200
        assert res.headers["x-harness-router"] == "disabled"
        models = {model for model, _ in fake.worker_calls}
        assert models == {get_settings().harness_worker_model}
    finally:
        get_settings.cache_clear()
        clear_frontier_cache()
