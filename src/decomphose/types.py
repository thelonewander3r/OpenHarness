from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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


class WorkerModelEntry(BaseModel):
    id: str
    tier: str
    cost_rank: int = Field(alias="costRank")
    notes: str | None = None

    model_config = {"populate_by_name": True}


class WorkerModelsConfig(BaseModel):
    """Tiered worker pool for per-micro-task routing (cheapest tier first)."""

    version: int
    updated_at: str = Field(alias="updatedAt")
    tiers: list[str]
    models: list[WorkerModelEntry]

    model_config = {"populate_by_name": True}


class MicroTask(BaseModel):
    id: str
    index: int = 0
    title: str
    instruction: str
    complexity: str = "standard"
    relevant_context_keys: list[str] = Field(
        default_factory=list, alias="relevantContextKeys"
    )

    model_config = {"populate_by_name": True}


TASK_COMPLEXITIES = ("routine", "standard", "complex")
DEFAULT_COMPLEXITY = "standard"


class DecomposedMicroTask(BaseModel):
    """Raw micro-task as emitted by the decomposition model, before normalization."""

    id: str | None = None
    title: str | None = None
    instruction: str
    complexity: str = DEFAULT_COMPLEXITY
    relevant_context_keys: list[str] = Field(
        default_factory=list, alias="relevantContextKeys"
    )

    model_config = {"populate_by_name": True}

    @field_validator("instruction")
    @classmethod
    def _instruction_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("instruction must be a non-empty string")
        return stripped

    @field_validator("complexity", mode="before")
    @classmethod
    def _normalize_complexity(cls, value: object) -> str:
        # A bad label must never fail the whole plan — coerce to the default tier.
        if isinstance(value, str) and value.strip().lower() in TASK_COMPLEXITIES:
            return value.strip().lower()
        return DEFAULT_COMPLEXITY


class DecompositionPlan(BaseModel):
    """Schema for the decomposition model's JSON output."""

    micro_tasks: list[DecomposedMicroTask] = Field(
        default_factory=list, alias="microTasks"
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
    models_used: list[str] | None = None
    decomposition_steps: int | None = None
    auditor_retries: int | None = None


class StrategyResult(BaseModel):
    body: Any = None
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    meta: StrategyResultMeta
    # Async iterator of SSE-encoded strings; when set, the server streams it and ignores body.
    stream: Any = None
