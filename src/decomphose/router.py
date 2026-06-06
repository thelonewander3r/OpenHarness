from __future__ import annotations

from decomphose.types import DEFAULT_COMPLEXITY, WorkerModelsConfig


def build_worker_escalation_path(
    config: WorkerModelsConfig | None,
    complexity: str,
    default_model: str,
) -> list[str]:
    """Ordered worker models for one micro-task: its tier first, then each tier above.

    The auditor retry loop walks this path — a rejection escalates to the next,
    more capable tier instead of re-asking the same model (Factory Router style).
    With no registry (config None) routing is disabled and the static default
    worker handles everything.
    """
    if config is None:
        return [default_model]

    tier = complexity if complexity in config.tiers else DEFAULT_COMPLEXITY
    if tier not in config.tiers:
        return [default_model]

    path: list[str] = []
    for candidate_tier in config.tiers[config.tiers.index(tier) :]:
        models = [m for m in config.models if m.tier == candidate_tier]
        if models:
            # Cheapest model that satisfies the tier.
            path.append(min(models, key=lambda m: m.cost_rank).id)

    return path or [default_model]
