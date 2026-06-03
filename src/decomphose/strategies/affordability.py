from __future__ import annotations

import json
import logging
import time
from typing import Any

from decomphose.strategies.context import StrategyContext
from decomphose.types import (
    AuditorVerdictPass,
    AuditorVerdictReject,
    ContextDocument,
    HarnessStrategy,
    MicroTask,
    StrategyResult,
    StrategyResultMeta,
)
from decomphose.utils.context_diet import extract_context_documents, slice_context_for_task
from decomphose.utils.errors import MicroTaskError
from decomphose.utils.logging import log_with_meta

log = logging.getLogger("decomphose.affordability")

DECOMP_SYSTEM = """You are a task decomposition engine. Given a complex user request and optional context documents, output ONLY valid JSON with this shape:
{
  "microTasks": [
    {
      "id": "task-1",
      "title": "short title",
      "instruction": "atomic instruction for one linear step",
      "relevantContextKeys": ["key-from-context-markers"]
    }
  ]
}
Rules:
- Break the master task into 3-8 sequential micro-tasks.
- Each micro-task must be independently executable in order.
- Use relevantContextKeys to reference [context:key] markers when present; use ["full-thread"] if no markers exist.
- Do not include markdown fences."""

WORKER_SYSTEM = (
    "You execute a single micro-task using ONLY the provided context slice. "
    "Be concise and factual. Output the result directly without meta commentary."
)

AUDITOR_SYSTEM = """You are a Goal Auditor. Evaluate whether the micro-task output satisfies its instruction.
Reply with EXACTLY one line:
PASS
or
REJECT: <specific feedback for retry>"""


async def execute_affordability_strategy(ctx: StrategyContext) -> StrategyResult:
    """Orchestrated Decomposition Sandbox."""
    messages: list[dict[str, Any]] = ctx.body.get("messages") or []
    user_prompt = _extract_primary_user_prompt(messages)
    context_docs = extract_context_documents(messages)
    settings = ctx.settings

    log_with_meta(
        log,
        logging.INFO,
        "Step 1 — Decomposition starting",
        {
            "requestId": ctx.meta.request_id,
            "decompModel": settings.harness_decomp_model,
            "contextChunks": len(context_docs),
        },
    )

    micro_tasks = await _decompose_master_task(
        ctx.client,
        settings.harness_decomp_model,
        user_prompt,
        context_docs,
    )

    log_with_meta(
        log,
        logging.INFO,
        "Decomposition complete",
        {
            "step": "decomposition",
            "microTaskCount": len(micro_tasks),
            "tasks": [{"id": t.id, "title": t.title} for t in micro_tasks],
        },
    )

    validated_outputs: list[str] = []
    total_auditor_retries = 0

    for task in micro_tasks:
        try:
            output, retries = await _run_micro_task_pipeline(
                ctx.client, settings, task, context_docs
            )
            total_auditor_retries += retries
            validated_outputs.append(f"## {task.title}\n\n{output}")
            log_with_meta(
                log,
                logging.INFO,
                "Micro-task validated",
                {"step": "micro-task-complete", "taskId": task.id, "auditorRetries": retries},
            )
        except MicroTaskError as exc:
            log_with_meta(
                log,
                logging.ERROR,
                "Micro-task failed (isolated)",
                {"taskId": task.id, "error": exc.message},
            )
            validated_outputs.append(
                f"## {task.title}\n\n[HARNESS: micro-task failed after retries — {exc.message}]"
            )

    log_with_meta(
        log,
        logging.INFO,
        "Step 4 — Compilation",
        {"step": "compilation", "segmentCount": len(validated_outputs)},
    )

    compiled = _compile_final_response(validated_outputs, user_prompt)
    response_body = _build_synthetic_completion(
        ctx.body.get("model") or "decomphose/affordability-compiled",
        compiled,
    )

    return StrategyResult(
        status=200,
        body=response_body,
        headers={
            "x-harness-strategy": HarnessStrategy.AFFORDABILITY.value,
            "x-harness-decomposition-steps": str(len(micro_tasks)),
            "x-harness-auditor-retries": str(total_auditor_retries),
        },
        meta=StrategyResultMeta(
            strategy=HarnessStrategy.AFFORDABILITY,
            model_used=settings.harness_worker_model,
            decomposition_steps=len(micro_tasks),
            auditor_retries=total_auditor_retries,
        ),
    )


async def _decompose_master_task(
    client: Any,
    model: str,
    user_prompt: str,
    context_docs: list[ContextDocument],
) -> list[MicroTask]:
    context_summary = "\n\n".join(
        f"[context:{d.key}]\n{d.content[:2000]}" for d in context_docs
    )

    raw = client.complete(
        model=model,
        messages=[
            {"role": "system", "content": DECOMP_SYSTEM},
            {
                "role": "user",
                "content": f"Master task:\n{user_prompt}\n\nContext documents:\n{context_summary}",
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    parsed = json.loads(raw)
    micro_tasks_raw = parsed.get("microTasks") or []
    if not micro_tasks_raw:
        return [
            MicroTask(
                id="task-1",
                index=0,
                title="Execute master task",
                instruction=user_prompt,
                relevant_context_keys=[d.key for d in context_docs],
            )
        ]

    result: list[MicroTask] = []
    for index, item in enumerate(micro_tasks_raw):
        result.append(
            MicroTask(
                id=item.get("id") or f"task-{index + 1}",
                index=index,
                title=item.get("title") or f"Step {index + 1}",
                instruction=item["instruction"],
                relevant_context_keys=item.get("relevantContextKeys") or [],
            )
        )
    return result


async def _run_micro_task_pipeline(
    client: Any,
    settings: Any,
    task: MicroTask,
    context_docs: list[ContextDocument],
) -> tuple[str, int]:
    context_slice = slice_context_for_task(context_docs, task)

    log_with_meta(
        log,
        logging.INFO,
        "Step 2 — Context diet applied",
        {
            "step": "context-diet",
            "taskId": task.id,
            "contextKeys": task.relevant_context_keys,
            "sliceChars": len(context_slice),
        },
    )

    last_feedback = ""
    retries = 0
    max_retries = settings.harness_max_auditor_retries

    for attempt in range(1, max_retries + 2):
        output = client.complete(
            model=settings.harness_worker_model,
            messages=[
                {"role": "system", "content": WORKER_SYSTEM},
                {
                    "role": "user",
                    "content": _build_worker_user_message(task, context_slice, last_feedback),
                },
            ],
            temperature=0.3,
        )

        log_with_meta(
            log,
            logging.INFO,
            "Step 3 — Goal auditor review",
            {"step": "goal-auditor", "taskId": task.id, "attempt": attempt},
        )

        verdict = _audit_micro_task_output(client, settings.harness_auditor_model, task, output)

        if isinstance(verdict, AuditorVerdictPass):
            log_with_meta(log, logging.INFO, "Auditor PASS", {"taskId": task.id, "attempt": attempt})
            return output, retries

        retries += 1
        last_feedback = verdict.feedback

        log_with_meta(
            log,
            logging.WARNING,
            "Auditor REJECT — retry scheduled",
            {
                "taskId": task.id,
                "attempt": attempt,
                "feedback": verdict.feedback,
                "retriesRemaining": max_retries - attempt,
            },
        )

        if attempt > max_retries:
            raise MicroTaskError(
                task.id,
                f"Goal auditor rejected after {max_retries} retries: {verdict.feedback}",
            )

    raise MicroTaskError(task.id, "Exhausted auditor retry loop")


def _audit_micro_task_output(
    client: Any,
    model: str,
    task: MicroTask,
    output: str,
) -> AuditorVerdictPass | AuditorVerdictReject:
    raw = client.complete(
        model=model,
        messages=[
            {"role": "system", "content": AUDITOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Micro-task instruction:\n{task.instruction}\n\n"
                    f"Output to review:\n{output}"
                ),
            },
        ],
        temperature=0.0,
    )

    trimmed = raw.strip()
    if trimmed.upper().startswith("PASS"):
        return AuditorVerdictPass()

    if trimmed.upper().startswith("REJECT"):
        feedback = trimmed.split(":", 1)[1].strip() if ":" in trimmed else trimmed
        return AuditorVerdictReject(feedback=feedback)

    return AuditorVerdictReject(feedback=trimmed)


def _build_worker_user_message(
    task: MicroTask,
    context_slice: str,
    retry_feedback: str,
) -> str:
    parts = [
        f"Micro-task: {task.title}",
        f"Instruction: {task.instruction}",
        f"Context slice:\n{context_slice}",
    ]
    if retry_feedback:
        parts.append(
            "Previous attempt was rejected by the Goal Auditor. Fix these issues:\n"
            f"{retry_feedback}"
        )
    return "\n\n".join(parts)


def _compile_final_response(segments: list[str], user_prompt: str) -> str:
    preview = user_prompt[:200] + ("…" if len(user_prompt) > 200 else "")
    header = f"Compiled response for: {preview}"
    return f"{header}\n\n" + "\n\n".join(segments)


def _build_synthetic_completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-harness-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _extract_primary_user_prompt(messages: list[dict[str, Any]]) -> str:
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""
    last = user_messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content)
