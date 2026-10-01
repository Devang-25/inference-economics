"""Workload trace generation.

A cost model is only as honest as its workload. Uniform-random prompts of equal
length make every optimization look great; real traffic is skewed in ways that
decide which optimizations actually pay:

  * Prompt/output lengths are heavy-tailed, not normal. The p99 request, not
    the mean, is what blows your KV budget and your tail latency.
  * Arrivals are bursty (Poisson with a diurnal multiplier), so queueing --
    not raw throughput -- drives p95 TTFT.
  * Prompts SHARE PREFIXES. A system prompt, a RAG context block, a multi-turn
    conversation. This is why prefix caching works at all.
  * Query popularity is Zipfian. A small head of questions covers a large share
    of traffic, which is where response caching earns its keep.
  * Difficulty is mixed. Most traffic is easy enough for a small model; a
    minority genuinely needs the big one. That's the routing opportunity.

Four workload archetypes, each with a different profile, because "it depends on
your workload" is the correct answer and this makes the dependence explicit.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    arrival_s: float
    prompt_text: str
    token_ids: list[int]
    output_tokens: int
    difficulty: float          # 0..1, ground truth used to score the router
    needs_big_model: bool      # ground truth: does the 8B answer acceptably?
    kind: str

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class Workload:
    name: str
    requests: list[Request]
    duration_s: float
    description: str = ""

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.requests)

    @property
    def arrival_rate_rps(self) -> float:
        return len(self.requests) / self.duration_s if self.duration_s else 0.0

    def stats(self) -> dict:
        pt = sorted(r.prompt_tokens for r in self.requests)
        ot = sorted(r.output_tokens for r in self.requests)
        def pct(xs, p):
            return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]
        return {
            "requests": len(self.requests),
            "duration_s": self.duration_s,
            "rps": self.arrival_rate_rps,
            "prompt_p50": pct(pt, 50), "prompt_p95": pct(pt, 95), "prompt_p99": pct(pt, 99),
            "output_p50": pct(ot, 50), "output_p95": pct(ot, 95),
            "total_prompt_tokens": sum(pt),
            "total_output_tokens": sum(ot),
            "pct_needs_big": 100.0 * sum(r.needs_big_model for r in self.requests) / len(self.requests),
        }


# -- archetypes ------------------------------------------------------------
@dataclass
class Archetype:
    kind: str
    weight: float
    system_prompt_tokens: int       # shared, cacheable prefix
    shared_context_tokens: int      # RAG chunks / file context, shared within a topic
    user_tokens_mu: float           # lognormal mu for the unique tail
    user_tokens_sigma: float
    output_mu: float
    output_sigma: float
    difficulty_alpha: float         # Beta(a,b): lower a => easier traffic
    difficulty_beta: float


ARCHETYPES = [
    # Chat: short prompts, long-ish answers, a fixed system prompt, easy traffic.
    Archetype("chat", 0.40, 180, 0, 3.8, 1.15, 5.85, 0.85, 2.0, 5.0),
    # RAG: large retrieved context, short answers, context reused across a topic.
    Archetype("rag", 0.30, 220, 900, 3.4, 1.0, 5.40, 0.75, 2.5, 4.0),
    # Agent: long growing scratchpad + tool schemas, short bursty outputs.
    Archetype("agent", 0.18, 700, 400, 4.4, 1.25, 5.25, 1.0, 3.5, 3.0),
    # Code: large file context, long generations, genuinely hard.
    Archetype("code", 0.12, 260, 600, 5.0, 1.15, 6.35, 0.95, 5.0, 2.5),
]

_HEAD_QUERIES = [
    "how do i reset my password", "what is your refund policy",
    "summarize this document", "explain this error message",
    "how do i cancel my subscription", "what are the pricing tiers",
    "write a unit test for this function", "translate this to spanish",
    "what changed in the latest release", "how do i add a teammate",
]


def _lognormal_int(rng: random.Random, mu: float, sigma: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(math.exp(rng.gauss(mu, sigma)))))


def generate(
    n_requests: int = 3000,
    duration_s: float = 120.0,
    seed: int = 20261001,
    burstiness: float = 1.8,
    head_share: float = 0.22,
    n_topics: int = 40,
    vocab: int = 128_000,
) -> Workload:
    """Generate a deterministic, production-shaped trace.

    `head_share` is the fraction of traffic drawn from a small Zipfian head of
    repeated queries -- the response-cache opportunity. `n_topics` controls how
    many distinct shared RAG/agent contexts exist -- the prefix-cache
    opportunity. Both are knobs precisely because both are workload properties
    you must measure on your own traffic before believing any cache number.
    """
    rng = random.Random(seed)
    weights = [a.weight for a in ARCHETYPES]

    # Stable token-id blocks for shared prefixes, so the prefix cache sees
    # genuinely identical token sequences (as a real tokenizer would emit).
    sys_prompts = {
        a.kind: [rng.randrange(vocab) for _ in range(a.system_prompt_tokens)]
        for a in ARCHETYPES
    }
    topics = {}
    for a in ARCHETYPES:
        if a.shared_context_tokens:
            topics[a.kind] = [
                [rng.randrange(vocab) for _ in range(a.shared_context_tokens)]
                for _ in range(n_topics)
            ]

    # Zipfian head of repeated queries.
    zipf_w = [1.0 / (i + 1) ** 1.1 for i in range(len(_HEAD_QUERIES))]

    requests: list[Request] = []
    t = 0.0
    base_rate = n_requests / duration_s
    for rid in range(n_requests):
        # Bursty arrivals: Poisson with a slow sinusoidal rate multiplier.
        phase = 2 * math.pi * (t / duration_s)
        rate = base_rate * (1.0 + (burstiness - 1.0) * 0.5 * (1 + math.sin(phase * 3)))
        t += rng.expovariate(max(rate, 1e-6))
        if t > duration_s:
            break

        arch = rng.choices(ARCHETYPES, weights=weights, k=1)[0]
        ids = list(sys_prompts[arch.kind])
        if arch.shared_context_tokens:
            ids += topics[arch.kind][rng.randrange(n_topics)]

        from_head = rng.random() < head_share
        if from_head:
            q = rng.choices(_HEAD_QUERIES, weights=zipf_w, k=1)[0]
            # Near-duplicate variants: the realistic source of cache hits.
            variant = rng.random()
            if variant < 0.35:
                text = q.capitalize() + "?"
            elif variant < 0.55:
                text = q.upper()
            elif variant < 0.70:
                text = "  " + q + " "
            else:
                text = q
            user_len = max(8, len(q) // 4)
        else:
            text = f"req-{rid}-{arch.kind}-{rng.randrange(1 << 30)}"
            user_len = _lognormal_int(rng, arch.user_tokens_mu, arch.user_tokens_sigma, 8, 12000)
        ids += [rng.randrange(vocab) for _ in range(user_len)]

        out = _lognormal_int(rng, arch.output_mu, arch.output_sigma, 8, 2048)
        diff = rng.betavariate(arch.difficulty_alpha, arch.difficulty_beta)
        # Ground truth: the 8B answers acceptably below a difficulty threshold.
        needs_big = diff > 0.55

        requests.append(Request(
            rid=rid, arrival_s=t, prompt_text=text, token_ids=ids,
            output_tokens=out, difficulty=diff, needs_big_model=needs_big,
            kind=arch.kind,
        ))

    # The nominal duration is an upper bound; burstiness means the trace
    # usually exhausts its request budget earlier. Bill against the span that
    # actually happened, not the one we asked for.
    actual_span = requests[-1].arrival_s if requests else duration_s
    return Workload(
        name="production-mix",
        requests=requests,
        duration_s=actual_span,
        description="40% chat / 30% RAG / 18% agent / 12% code, bursty Poisson arrivals",
    )
