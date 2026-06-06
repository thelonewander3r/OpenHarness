"""Tests for parallel micro-task execution (dependency waves)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi.testclient import TestClient

from decomphose.server import create_app
from decomphose.utils.decomposition import parse_decomposition_plan

REQUEST_BODY: dict[str, Any] = {
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Do the master task."}],
}


# --- dependency sanitization unit tests ---


def plan_json(tasks: list[dict[str, Any]]) -> str:
    return json.dumps({"microTasks": tasks})


def test_valid_backward_dependency_kept() -> None:
    tasks = parse_decomposition_plan(
        plan_json(
            [
                {"id": "task-1", "instruction": "a"},
                {"id": "task-2", "instruction": "b", "dependsOn": ["task-1"]},
            ]
        )
    )
    assert tasks[1].depends_on == ["task-1"]


def test_forward_self_and_unknown_deps_dropped() -> None:
    tasks = parse_decomposition_plan(
        plan_json(
            [
                {"id": "task-1", "instruction": "a", "dependsOn": ["task-2", "task-1", "ghost"]},
                {"id": "task-2", "instruction": "b", "dependsOn": ["task-2", "nope"]},
            ]
        )
    )
    assert tasks[0].depends_on == []
    assert tasks[1].depends_on == []


def test_malformed_depends_on_coerced_to_independent() -> None:
    tasks = parse_decomposition_plan(
        plan_json([{"id": "task-1", "instruction": "a", "dependsOn": "task-0"}])
    )
    assert tasks[0].depends_on == []


def test_duplicate_task_ids_deduplicated() -> None:
    tasks = parse_decomposition_plan(
        plan_json(
            [
                {"id": "task-1", "instruction": "a"},
                {"id": "task-1", "instruction": "b"},
            ]
        )
    )
    assert tasks[0].id != tasks[1].id


# --- end-to-end wave scheduling ---


class ParallelFakeClient:
    """Tracks worker concurrency and captures worker prompts."""

    def __init__(self, plan: str) -> None:
        self._plan = plan
        self.active_workers = 0
        self.max_active_workers = 0
        self.worker_messages: dict[str, str] = {}  # instruction line -> full message

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
            return self._plan
        if "Goal Auditor" in system:
            return "PASS"

        user = messages[1]["content"]
        self.active_workers += 1
        self.max_active_workers = max(self.max_active_workers, self.active_workers)
        await asyncio.sleep(0.05)  # Hold the slot so overlap is observable.
        self.active_workers -= 1
        title = user.splitlines()[0].removeprefix("Micro-task: ")
        self.worker_messages[title] = user
        return f"output of {title}"

    async def close(self) -> None:
        pass


def post_affordability(fake: ParallelFakeClient):
    with TestClient(create_app(client=fake)) as client:
        return client.post(
            "/v1/chat/completions",
            json=REQUEST_BODY,
            headers={"X-Harness-Strategy": "affordability"},
        )


def test_independent_tasks_run_concurrently() -> None:
    plan = plan_json(
        [
            {"id": f"task-{i}", "title": f"T{i}", "instruction": f"do {i}"}
            for i in range(1, 4)
        ]
    )
    fake = ParallelFakeClient(plan)
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-parallel-waves"] == "1"
    assert fake.max_active_workers >= 2  # genuinely overlapping, not sequential


def test_dependent_task_waits_and_receives_prerequisite_output() -> None:
    plan = plan_json(
        [
            {"id": "task-1", "title": "First", "instruction": "do first"},
            {
                "id": "task-2",
                "title": "Second",
                "instruction": "do second",
                "dependsOn": ["task-1"],
            },
        ]
    )
    fake = ParallelFakeClient(plan)
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-parallel-waves"] == "2"
    assert fake.max_active_workers == 1  # chain cannot overlap
    second_message = fake.worker_messages["Second"]
    assert "Outputs from prerequisite steps" in second_message
    assert "output of First" in second_message


def test_diamond_graph_runs_in_three_waves() -> None:
    plan = plan_json(
        [
            {"id": "task-1", "title": "Root", "instruction": "root"},
            {"id": "task-2", "title": "LeftBranch", "instruction": "l", "dependsOn": ["task-1"]},
            {"id": "task-3", "title": "RightBranch", "instruction": "r", "dependsOn": ["task-1"]},
            {
                "id": "task-4",
                "title": "Merge",
                "instruction": "m",
                "dependsOn": ["task-2", "task-3"],
            },
        ]
    )
    fake = ParallelFakeClient(plan)
    res = post_affordability(fake)

    assert res.status_code == 200
    assert res.headers["x-harness-parallel-waves"] == "3"
    assert fake.max_active_workers == 2  # the two branches overlap
    merge_message = fake.worker_messages["Merge"]
    assert "output of LeftBranch" in merge_message
    assert "output of RightBranch" in merge_message
    # Compilation preserves decomposition order.
    content = res.json()["choices"][0]["message"]["content"]
    assert content.index("## Root") < content.index("## LeftBranch") < content.index("## Merge")


def test_failed_dependency_does_not_block_dependent() -> None:
    class FailingFirstClient(ParallelFakeClient):
        async def complete(self, *, model, messages, **kwargs):
            system = messages[0]["content"]
            if "Goal Auditor" in system and "do first" in messages[1]["content"]:
                return "REJECT: hopeless"
            return await super().complete(model=model, messages=messages, **kwargs)

    plan = plan_json(
        [
            {"id": "task-1", "title": "First", "instruction": "do first"},
            {
                "id": "task-2",
                "title": "Second",
                "instruction": "do second",
                "dependsOn": ["task-1"],
            },
        ]
    )
    fake = FailingFirstClient(plan)
    res = post_affordability(fake)

    assert res.status_code == 200
    content = res.json()["choices"][0]["message"]["content"]
    assert "[HARNESS: micro-task failed" in content
    assert "output of Second" in content  # dependent still ran
    # Failed prerequisite output is not fed to the dependent.
    assert "Outputs from prerequisite steps" not in fake.worker_messages["Second"]
