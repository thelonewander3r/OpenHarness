from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from decomphose.types import ModelPricingConfig


@dataclass
class UsageEntry:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None  # None when the model has no pricing entry


@dataclass
class UsageLedger:
    """Per-request accumulator of upstream token usage and estimated cost."""

    pricing: ModelPricingConfig | None = None
    entries: list[UsageEntry] = field(default_factory=list)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.entries.append(
            UsageEntry(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=self._cost_for(model, prompt_tokens, completion_tokens),
            )
        )

    def _cost_for(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        if self.pricing is None:
            return None
        entry = next((p for p in self.pricing.models if p.id == model), None)
        if entry is None:
            return None
        return (
            prompt_tokens / 1_000_000 * entry.prompt_per_mtok
            + completion_tokens / 1_000_000 * entry.completion_per_mtok
        )

    @property
    def llm_calls(self) -> int:
        return len(self.entries)

    @property
    def prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.entries)

    @property
    def completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.entries)

    @property
    def cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries if e.cost_usd is not None)

    @property
    def coverage(self) -> str:
        """How much of the spend is priced: full | partial | none."""
        priced = [e for e in self.entries if e.cost_usd is not None]
        if not priced:
            return "none"
        return "full" if len(priced) == len(self.entries) else "partial"

    def headers(self) -> dict[str, str]:
        if not self.entries:
            return {}
        headers = {
            "x-harness-llm-calls": str(self.llm_calls),
            "x-harness-prompt-tokens": str(self.prompt_tokens),
            "x-harness-completion-tokens": str(self.completion_tokens),
            "x-harness-cost-coverage": self.coverage,
        }
        if self.coverage != "none":
            headers["x-harness-cost-usd"] = f"{self.cost_usd:.6f}"
        return headers


_ledger_var: ContextVar[UsageLedger | None] = ContextVar(
    "decomphose_usage_ledger", default=None
)


def start_ledger(pricing: ModelPricingConfig | None) -> UsageLedger:
    """Begin a fresh ledger for the current request context."""
    ledger = UsageLedger(pricing=pricing)
    _ledger_var.set(ledger)
    return ledger


def current_ledger() -> UsageLedger | None:
    return _ledger_var.get()
