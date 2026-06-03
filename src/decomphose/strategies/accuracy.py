from __future__ import annotations

import logging
from typing import Any

from decomphose.config import load_frontier_models
from decomphose.strategies.context import StrategyContext
from decomphose.types import FrontierModelEntry, HarnessStrategy, StrategyResult, StrategyResultMeta
from decomphose.utils.context_diet import (
    estimate_token_budget,
    extract_context_documents,
)
from decomphose.utils.logging import log_with_meta

log = logging.getLogger("decomphose.accuracy")


async def execute_accuracy_strategy(ctx: StrategyContext) -> StrategyResult:
    """Precision Router: highest-capability frontier model that fits context budget."""
    messages: list[dict[str, Any]] = ctx.body.get("messages") or []
    context_docs = extract_context_documents(messages)
    context_text = "\n".join(d.content for d in context_docs)
    estimated_tokens = estimate_token_budget(context_text)

    log_with_meta(
        log,
        logging.INFO,
        "Analyzing prompt and context",
        {
            "requestId": ctx.meta.request_id,
            "messageCount": len(messages),
            "contextChunks": len(context_docs),
            "estimatedTokens": estimated_tokens,
        },
    )

    frontier = load_frontier_models()
    selected = _select_frontier_model(frontier.models, estimated_tokens)

    log_with_meta(
        log,
        logging.INFO,
        "Routing to frontier model",
        {
            "model": selected.id,
            "capabilityRank": selected.capability_rank,
            "contextWindow": selected.context_window,
        },
    )

    routed_body = {**ctx.body, "model": selected.id}
    completion = ctx.client.forward_raw(routed_body)

    return StrategyResult(
        status=200,
        body=completion.model_dump(),
        headers={
            "x-harness-selected-model": selected.id,
            "x-harness-strategy": HarnessStrategy.ACCURACY.value,
        },
        meta=StrategyResultMeta(
            strategy=HarnessStrategy.ACCURACY,
            model_used=selected.id,
        ),
    )


def _select_frontier_model(
    models: list[FrontierModelEntry],
    estimated_tokens: int,
) -> FrontierModelEntry:
    sorted_models = sorted(models, key=lambda m: m.capability_rank, reverse=True)
    fitting = [m for m in sorted_models if m.context_window >= estimated_tokens]
    return fitting[0] if fitting else sorted_models[0]
