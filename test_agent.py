#!/usr/bin/env python3
"""
Simulates an autonomous agent framework sending a heavy task through Decomphose.

  python test_agent.py
  MOCK_AFFORDABILITY=1 python test_agent.py
"""

from __future__ import annotations

import json
import os
import sys

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

HARNESS_URL = os.environ.get("HARNESS_URL", "http://127.0.0.1:3100")
MOCK = os.environ.get("MOCK_AFFORDABILITY") == "1"

HEAVY_TASK = {
    "model": "openai/gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": "You are an autonomous coding agent. Complete all sub-goals thoroughly.",
        },
        {
            "role": "user",
            "content": """Master task: Design and document a minimal event-sourced inventory microservice.

Requirements:
1. Define aggregate boundaries and command handlers.
2. Specify read-model projections for stock levels.
3. Outline idempotent event publishing to a message bus.
4. Provide a failure-mode table (split-brain, duplicate delivery, stale reads).

[context:domain-brief]
Event-sourced inventory tracks SKU stock as immutable events. Commands: ReserveStock, ReleaseReservation, AdjustStock. Events: StockReserved, ReservationReleased, StockAdjusted.

[context:constraints]
Must run on Node 20+, PostgreSQL for event store, no Kafka in v1 — use PostgreSQL NOTIFY for outbox pattern.""",
        },
    ],
    "temperature": 0.2,
}


def run_mock_pipeline_log() -> None:
    print("\n=== MOCK Affordability Pipeline (console trace only) ===\n")

    micro_tasks = [
        {
            "id": "task-1",
            "title": "Aggregate boundaries",
            "instruction": "Define inventory aggregate roots and command handlers.",
            "relevantContextKeys": ["domain-brief"],
        },
        {
            "id": "task-2",
            "title": "Read models",
            "instruction": "Specify projections for current stock levels.",
            "relevantContextKeys": ["domain-brief", "constraints"],
        },
        {
            "id": "task-3",
            "title": "Outbox publishing",
            "instruction": "Outline idempotent event publishing via PostgreSQL NOTIFY.",
            "relevantContextKeys": ["constraints"],
        },
        {
            "id": "task-4",
            "title": "Failure modes",
            "instruction": "Produce failure-mode table for split-brain, duplicates, stale reads.",
            "relevantContextKeys": ["domain-brief", "constraints"],
        },
    ]

    print("[affordability] Step 1 — Decomposition complete")
    print(json.dumps([{"id": t["id"], "title": t["title"]} for t in micro_tasks], indent=2))

    for task in micro_tasks:
        print(f"\n[affordability] Step 2 — Context diet: {task['id']}")
        print(f"  keys: {', '.join(task['relevantContextKeys'])}")

        attempt = 0
        passed = False
        while attempt < 3 and not passed:
            attempt += 1
            print(f"[affordability] Step 3 — Goal auditor: {task['id']} attempt {attempt}")

            if task["id"] == "task-3" and attempt == 1:
                print("  verdict: REJECT — missing idempotency key description")
                print("  action: retry with auditor feedback")
                continue

            print("  verdict: PASS")
            passed = True

        print(f"[affordability] Micro-task complete: {task['id']}")

    print("\n[affordability] Step 4 — Compilation")
    print("  merged 4 validated segments into synthetic chat.completion\n")


def run_live_affordability() -> None:
    if httpx is None:
        print("Install httpx: pip install decomphose[dev]")
        sys.exit(1)

    print("\n=== Decomphose Test Agent — AFFORDABILITY ===\n")
    print(f"POST {HARNESS_URL}/v1/chat/completions")
    print("Header: X-Harness-Strategy: affordability\n")

    with httpx.Client(timeout=300.0) as client:
        res = client.post(
            f"{HARNESS_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "X-Harness-Strategy": "affordability",
            },
            json=HEAVY_TASK,
        )

    print("Response status:", res.status_code)
    print(
        "Harness headers:",
        {
            "strategy": res.headers.get("x-harness-strategy"),
            "steps": res.headers.get("x-harness-decomposition-steps"),
            "retries": res.headers.get("x-harness-auditor-retries"),
            "requestId": res.headers.get("x-harness-request-id"),
        },
    )

    data = res.json()
    if "error" in data:
        print("Error:", data["error"].get("message"))
        return

    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    print("\n--- Compiled output (excerpt) ---\n")
    print(content[:1500])
    if len(content) > 1500:
        print("\n... [truncated]")


def main() -> None:
    if MOCK:
        run_mock_pipeline_log()
        return

    if httpx is None:
        print("Install httpx: pip install decomphose[dev]")
        sys.exit(1)

    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(f"{HARNESS_URL}/health")
            health.raise_for_status()
    except Exception:
        print(f"Harness not reachable at {HARNESS_URL}. Start with:")
        print("  pip install -e . && decomphose")
        print("Or mock trace: MOCK_AFFORDABILITY=1 python test_agent.py")
        sys.exit(1)

    run_live_affordability()


if __name__ == "__main__":
    main()
