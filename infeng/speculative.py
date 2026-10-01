"""Speculative decoding.

A small draft model proposes k tokens; the big model verifies all k+1 positions
in ONE forward pass and accepts the longest correct prefix. Output is provably
identical in distribution to sampling from the target model -- this is a pure
latency/throughput optimization, not a quality trade.

The economics are subtler than the marketing:

  * Decode is memory-bound, so verifying k+1 positions costs barely more than
    verifying 1 -- the weights get read once either way. That's the free lunch.
  * But the FLOPs scale with (k+1) x batch. At LOW batch you are memory-bound
    and speculation is nearly free. At HIGH batch you are already compute-bound,
    and speculation now competes for the same FLOPs it needs -- so the win
    shrinks and can go NEGATIVE.
  * Acceptance is workload-dependent. Boilerplate, structured output and code
    accept at a high rate; genuinely novel reasoning accepts poorly.

So speculative decoding is a LATENCY tool that helps most when you are latency-
constrained at low batch, and it is the first thing to turn off when you are
throughput-constrained. The simulator models both sides so the crossover shows
up rather than being asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

from .roofline import Executor


@dataclass
class SpecConfig:
    k: int = 4                        # draft tokens proposed per step
    base_acceptance: float = 0.82     # alpha for the easiest traffic
    difficulty_penalty: float = 0.45  # alpha falls this much across difficulty 0->1
    enabled: bool = True

    def acceptance_for(self, difficulty: float) -> float:
        a = self.base_acceptance - self.difficulty_penalty * difficulty
        return max(0.05, min(0.98, a))


def expected_accepted(alpha: float, k: int) -> float:
    """Expected tokens emitted per verification step.

    Draft tokens are accepted while they match, so the number accepted is
    geometric: sum_{i=1..k} alpha^i. The target model always contributes one
    correct token from the verification pass itself, hence the +1.
    """
    if alpha >= 1.0:
        return k + 1.0
    geom = alpha * (1.0 - alpha ** k) / (1.0 - alpha)
    return geom + 1.0


@dataclass
class SpeculativeDecoder:
    target: Executor
    draft: Executor
    cfg: SpecConfig

    def step_time_and_tokens(
        self, batch_size: int, kv_tokens: float, mean_alpha: float
    ) -> tuple[float, float]:
        """Returns (seconds for one speculative iteration, mean tokens emitted
        per sequence). Compare against target.decode_step_time_s(batch, kv)
        which emits exactly 1.0."""
        k = self.cfg.k
        # Draft runs k sequential forward passes on the small model.
        draft_s = sum(
            self.draft.decode_step_time_s(batch_size, kv_tokens * (self.draft.model.layers /
                                                                  self.target.model.layers))
            for _ in range(k)
        )
        # Target verifies k+1 positions per sequence in one pass: weights read
        # once (memory term unchanged), FLOPs scale with batch*(k+1).
        verify_s = self.target.decode_step_time_s(batch_size * (k + 1), kv_tokens)
        emitted = expected_accepted(mean_alpha, k)
        return draft_s + verify_s, emitted

    def effective_tpot_s(self, batch_size: int, kv_tokens: float, mean_alpha: float) -> float:
        dt, emitted = self.step_time_and_tokens(batch_size, kv_tokens, mean_alpha)
        return dt / max(emitted, 1e-9)
