from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HarnessStrategy(str, Enum):
    ACCURACY = "accuracy"
    AFFORDABILITY = "affordability"


HARNESS_STRATEGY_HEADER = "x-harness-strategy"


class HarnessRequestMeta(BaseModel):
    strategy: HarnessStrategy
    request_id: str


class FrontierModelEntry(BaseModel):
    id: str
    provider: str
    capability_rank: int = Field(alias="capabilityRank")
    context_window: int = Field(alias="contextWindow")
    notes: str | None = None

    model_config = {"populate_by_name": True}


class FrontierModelsConfig(BaseModel):
    version: int
    updated_at: str = Field(alias="updatedAt")
    models: list[FrontierModelEntry]

    model_config = {"populate_by_name": True}


class MicroTask(BaseModel):
    id: str
    index: int = 0
    title: str
    instruction: str
    relevant_context_keys: list[str] = Field(
        default_factory=list, alias="relevantContextKeys"
    )

    model_config = {"populate_by_name": True}


class ContextDocument(BaseModel):
    key: str
    content: str


class AuditorVerdictPass(BaseModel):
    status: Literal["pass"] = "pass"
    feedback: str | None = None


class AuditorVerdictReject(BaseModel):
    status: Literal["reject"] = "reject"
    feedback: str


AuditorVerdict = AuditorVerdictPass | AuditorVerdictReject


class StrategyResultMeta(BaseModel):
    strategy: HarnessStrategy
    model_used: str | None = None
    decomposition_steps: int | None = None
    auditor_retries: int | None = None


class StrategyResult(BaseModel):
    body: Any = None
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    meta: StrategyResultMeta
    # Async iterator of SSE-encoded strings; when set, the server streams it and ignores body.
    stream: Any = None
