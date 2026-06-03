from __future__ import annotations

from decomphose.strategies.accuracy import execute_accuracy_strategy
from decomphose.strategies.affordability import execute_affordability_strategy
from decomphose.strategies.context import StrategyContext
from decomphose.types import HarnessStrategy, StrategyResult


async def dispatch_strategy(ctx: StrategyContext) -> StrategyResult:
    if ctx.meta.strategy == HarnessStrategy.ACCURACY:
        return await execute_accuracy_strategy(ctx)
    if ctx.meta.strategy == HarnessStrategy.AFFORDABILITY:
        return await execute_affordability_strategy(ctx)
    raise ValueError(f"Unknown strategy: {ctx.meta.strategy}")
