from __future__ import annotations

import json
import logging
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from decomphose.strategies.context import StrategyContext
from decomphose.telemetry import get_tracer
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
from decomphose.utils.decomposition import parse_decomposition_plan
from decomphose.utils.errors import DecompositionError, MicroTaskError
from decomphose.utils.logging import log_with_meta
from decomphose.utils.streaming import sse_synthetic_completion

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

    with get_tracer().start_as_current_span("affordability.decompose") as span:
        span.set_attribute("llm.model", settings.harness_decomp_model)
        micro_tasks = await _decompose_master_task(
            ctx.client,
            settings.harness_decomp_model,
            user_prompt,
            context_docs,
        )
        span.set_attribute("harness.micro_task_count", len(micro_tasks))

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
        with get_tracer().start_as_current_span("affordability.micro_task") as span:
            span.set_attribute("harness.task.id", task.id)
            span.set_attribute("harness.task.title", task.title)
            span.set_attribute("harness.task.context_keys", ",".join(task.relevant_context_keys))
            try:
                output, retries = await _run_micro_task_pipeline(
                    ctx.client, settings, task, context_docs
                )
                total_auditor_retries += retries
                span.set_attribute("harness.auditor_retries", retries)
                validated_outputs.append(f"## {task.title}\n\n{output}")
                log_with_meta(
                    log,
                    logging.INFO,
                    "Micro-task validated",
                    {"step": "micro-task-complete", "taskId": task.id, "auditorRetries": retries},
                )
            except MicroTaskError as exc:
                span.set_status(StatusCode.ERROR, exc.message)
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
    model = ctx.body.get("model") or "decomphose/affordability-compiled"
    headers = {
        "x-harness-strategy": HarnessStrategy.AFFORDABILITY.value,
        "x-harness-decomposition-steps": str(len(micro_tasks)),
        "x-harness-auditor-retries": str(total_auditor_retries),
    }
    meta = StrategyResultMeta(
        strategy=HarnessStrategy.AFFORDABILITY,
        model_used=settings.harness_worker_model,
        decomposition_steps=len(micro_tasks),
        auditor_retries=total_auditor_retries,
    )

    if ctx.body.get("stream"):
        # The pipeline already ran to completion; replay the compiled result as SSE chunks
        # so streaming clients work unchanged.
        return StrategyResult(
            status=200,
            stream=sse_synthetic_completion(
                completion_id=f"chatcmpl-harness-{int(time.time() * 1000)}",
                created=int(time.time()),
                model=model,
                content=compiled,
            ),
            headers=headers,
            meta=meta,
        )

    return StrategyResult(
        status=200,
        body=_build_synthetic_completion(model, compiled),
        headers=headers,
        meta=meta,
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

    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": DECOMP_SYSTEM},
        {
            "role": "user",
            "content": f"Master task:\n{user_prompt}\n\nContext documents:\n{context_summary}",
        },
    ]

    messages = base_messages
    for attempt in (1, 2):
        raw = await client.complete(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        try:
            micro_tasks = parse_decomposition_plan(raw)
        except DecompositionError as exc:
            log_with_meta(
                log,
                logging.WARNING,
                "Decomposition output invalid",
                {"attempt": attempt, "error": exc.message},
            )
            # One corrective retry: feed the bad output and the validation error back.
            messages = [
                *base_messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That response was invalid: {exc.message}\n"
                        "Respond again with ONLY the corrected JSON object — no fences, no prose."
                    ),
                },
            ]
            continue

        if micro_tasks:
            return micro_tasks
        break  # Valid but empty plan — fall through to single-task fallback.

    log_with_meta(
        log,
        logging.WARNING,
        "Decomposition failed validation — falling back to single master task",
        {"decompModel": model},
    )
    return [_fallback_micro_task(user_prompt, context_docs)]


def _fallback_micro_task(
    user_prompt: str,
    context_docs: list[ContextDocument],
) -> MicroTask:
    return MicroTask(
        id="task-1",
        index=0,
        title="Execute master task",
        instruction=user_prompt,
        relevant_context_keys=[d.key for d in context_docs],
    )


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
        output = await client.complete(
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

        verdict = await _audit_micro_task_output(
            client, settings.harness_auditor_model, task, output
        )

        if isinstance(verdict, AuditorVerdictPass):
            trace.get_current_span().add_event("auditor.pass", {"attempt": attempt})
            log_with_meta(log, logging.INFO, "Auditor PASS", {"taskId": task.id, "attempt": attempt})
            return output, retries

        retries += 1
        last_feedback = verdict.feedback
        trace.get_current_span().add_event(
            "auditor.reject", {"attempt": attempt, "feedback": verdict.feedback}
        )

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


async def _audit_micro_task_output(
    client: Any,
    model: str,
    task: MicroTask,
    output: str,
) -> AuditorVerdictPass | AuditorVerdictReject:
    raw = await client.complete(
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
