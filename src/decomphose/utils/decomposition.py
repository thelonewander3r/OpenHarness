from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from decomphose.types import DecompositionPlan, MicroTask
from decomphose.utils.errors import DecompositionError
from decomphose.utils.logging import log_with_meta

log = logging.getLogger("decomphose.decomposition")

# Hard cap matching the decomposition prompt's 3-8 contract — bounds worker/auditor cost
# even if the model over-decomposes.
MAX_MICRO_TASKS = 8


def strip_json_fences(raw: str) -> str:
    """Remove markdown code fences some models wrap around JSON despite instructions."""
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_decomposition_plan(raw: str) -> list[MicroTask]:
    """Parse and validate decomposition model output into normalized MicroTasks.

    Raises DecompositionError on malformed JSON or schema violations; an empty
    (but valid) plan returns [] so the caller can apply its fallback.
    """
    text = strip_json_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecompositionError(
            f"Decomposition output is not valid JSON: {exc.msg} (line {exc.lineno})",
            cause=exc,
        ) from exc

    try:
        plan = DecompositionPlan.model_validate(payload)
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:3]
        )
        raise DecompositionError(
            f"Decomposition JSON failed schema validation: {issues}",
            cause=exc,
        ) from exc

    if len(plan.micro_tasks) > MAX_MICRO_TASKS:
        log_with_meta(
            log,
            logging.WARNING,
            "Decomposition over-decomposed — truncating",
            {"produced": len(plan.micro_tasks), "kept": MAX_MICRO_TASKS},
        )

    return [
        MicroTask(
            id=item.id or f"task-{index + 1}",
            index=index,
            title=item.title or f"Step {index + 1}",
            instruction=item.instruction,
            relevant_context_keys=item.relevant_context_keys,
        )
        for index, item in enumerate(plan.micro_tasks[:MAX_MICRO_TASKS])
    ]
