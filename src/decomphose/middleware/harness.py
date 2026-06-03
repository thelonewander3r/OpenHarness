from __future__ import annotations

import uuid

from fastapi import Request

from decomphose.types import HARNESS_STRATEGY_HEADER, HarnessRequestMeta, HarnessStrategy
from decomphose.utils.errors import HarnessError


def parse_harness_strategy(request: Request) -> HarnessRequestMeta:
    raw = request.headers.get(HARNESS_STRATEGY_HEADER)
    if not raw:
        raise HarnessError(
            f"Missing required header: {HARNESS_STRATEGY_HEADER}",
            "MISSING_STRATEGY",
            400,
        )

    normalized = raw.lower().strip()
    try:
        strategy = HarnessStrategy(normalized)
    except ValueError as exc:
        raise HarnessError(
            f'Invalid {HARNESS_STRATEGY_HEADER}: must be "accuracy" or "affordability"',
            "INVALID_STRATEGY",
            400,
        ) from exc

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    return HarnessRequestMeta(strategy=strategy, request_id=request_id)
