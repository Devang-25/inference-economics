"""Cost-aware model routing.

The largest single lever in this whole talk, and the one with the least
engineering glamour: most requests do not need your biggest model. If 70% of
traffic is answered acceptably by an 8B, routing it there is a ~9x cost
reduction on that share -- larger than any kernel-level optimization -- for the
price of a classifier that runs in microseconds.

The classifier here is deliberately boring and deliberately CHEAP: hand-built
lexical and structural features with a linear score, no model call. Spending an
LLM call to decide whether to make an LLM call is a tax on every request, and
it puts a network dependency on your critical path.

The part that matters is not the classifier -- it's the CALIBRATION. A router
is a precision/recall trade, and the two errors are not symmetric:

    Under-route (hard question -> small model)  = a bad answer. Expensive.
    Over-route  (easy question -> big model)    = a correct answer. Just costs money.

So we tune the threshold to bound the under-route rate, accept some waste, and
report BOTH numbers instead of only the flattering one. An escalation path
(small model answers, a cheap check escalates on low confidence) converts
remaining under-routes into latency rather than into wrong answers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .trace import Request

_HARD_MARKERS = (
    "prove", "derive", "analyze", "analyse", "refactor", "architect", "design",
    "optimize", "optimise", "debug", "trade-off", "tradeoff", "compare",
    "why does", "step by step", "reason", "algorithm", "complexity", "migrate",
)
_EASY_MARKERS = (
    "what is", "what are", "how do i", "how can i", "list", "translate",
    "summarize", "summarise", "define", "when is", "where is", "who is",
    "cancel", "reset", "pricing", "refund",
)
_CODE = re.compile(r"```|def |class |import |SELECT |function |=>|\{\s*\"")


@dataclass
class RouterStats:
    routed_small: int = 0
    routed_big: int = 0
    under_routed: int = 0    # needed big, sent small -> quality incident
    over_routed: int = 0     # fine on small, sent big -> wasted money
    escalated: int = 0

    @property
    def total(self) -> int:
        return self.routed_small + self.routed_big

    @property
    def small_share_pct(self) -> float:
        return 100.0 * self.routed_small / self.total if self.total else 0.0

    @property
    def under_route_pct(self) -> float:
        return 100.0 * self.under_routed / self.total if self.total else 0.0

    @property
    def over_route_pct(self) -> float:
        return 100.0 * self.over_routed / self.total if self.total else 0.0


@dataclass
class DifficultyRouter:
    """Score in [0,1]; above `threshold` goes to the big model."""
    threshold: float = 0.42
    escalate: bool = True
    # Ground-truth leakage knob: the simulator knows each request's true
    # difficulty, so we model a realistic classifier as "true difficulty
    # observed through noise" rather than pretending to a perfect oracle.
    classifier_noise: float = 0.14
    seed: int = 7
    stats: RouterStats = field(default_factory=RouterStats)

    def features(self, req: Request) -> dict:
        t = req.prompt_text.lower()
        return {
            "len": min(req.prompt_tokens / 4000.0, 1.0),
            "hard": sum(m in t for m in _HARD_MARKERS),
            "easy": sum(m in t for m in _EASY_MARKERS),
            "code": 1.0 if _CODE.search(req.prompt_text) else 0.0,
            "kind_code": 1.0 if req.kind == "code" else 0.0,
            "kind_agent": 1.0 if req.kind == "agent" else 0.0,
        }

    def score(self, req: Request) -> float:
        f = self.features(req)
        z = (
            -0.35
            + 1.10 * f["len"]
            + 0.55 * f["hard"]
            - 0.65 * f["easy"]
            + 0.60 * f["code"]
            + 0.85 * f["kind_code"]
            + 0.45 * f["kind_agent"]
        )
        lexical = 1.0 / (1.0 + math.exp(-z))
        # Blend the observable lexical signal with a noisy view of the request's
        # true difficulty. This is what a trained classifier approximates; the
        # noise term is what stops the demo from claiming a perfect router.
        noise = ((hash((req.rid, self.seed)) % 10_000) / 10_000.0 - 0.5) * 2.0
        observed = req.difficulty + noise * self.classifier_noise
        return max(0.0, min(1.0, 0.45 * lexical + 0.55 * observed))

    def route(self, req: Request) -> str:
        s = self.score(req)
        tier = "big" if s >= self.threshold else "small"
        if tier == "small":
            self.stats.routed_small += 1
            if req.needs_big_model:
                self.stats.under_routed += 1
                if self.escalate:
                    self.stats.escalated += 1
        else:
            self.stats.routed_big += 1
            if not req.needs_big_model:
                self.stats.over_routed += 1
        return tier

    def quality_delta_pct(self, small_model_gap_pct: float = -3.5) -> float:
        """Blended quality delta vs always-big.

        Only UNDER-routed requests degrade. With escalation on, a low-confidence
        small-model answer is retried on the big model, so most under-routes
        become extra latency and cost rather than a worse answer -- we model an
        85% catch rate, which is what a cheap verifier realistically gets.
        """
        if not self.stats.total:
            return 0.0
        bad = self.stats.under_routed
        if self.escalate:
            bad *= 0.15
        return small_model_gap_pct * (bad / self.stats.total)
