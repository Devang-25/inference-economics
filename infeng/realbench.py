"""Real measured API economics — the ground truth under the simulator.

Everything else in this repo is a model. This module is not: it makes real
Anthropic API calls and records what actually happened, so two of the
simulator's claims can be checked against reality rather than asserted.

  1. PREFIX CACHING. The simulator says re-sending a shared prefix should be
     nearly free. Anthropic's prompt caching bills cache reads at ~0.1x input
     rate, and `usage.cache_read_input_tokens` reports exactly how many tokens
     were served from cache. So we send the same large system prompt twice and
     read the real numbers off the response. Same mechanism as a KV prefix
     cache in your own server, with a receipt attached.

  2. MODEL ROUTING. The simulator says routing easy traffic to a small model is
     the biggest single lever. We send the same prompts to a small and a large
     model and measure real latency and real cost at published rates.

Record once on good Wi-Fi, replay forever offline:

    python -m infeng.realbench record     # a few cents, ~8 short calls
    python -m infeng.realbench show       # replay the cassette, no network

The cassette is committed, so the stage demo never touches the network and
anyone cloning the repo sees the same measurements without paying for them.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CASSETTE = Path(__file__).resolve().parent.parent / "bench" / "realbench_cassette.json"

# Published list prices, USD per million tokens. Cache reads bill at 0.1x input,
# cache writes at 1.25x input (5-minute TTL).
PRICES = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-opus-5": {"in": 5.00, "out": 25.00},
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25

SMALL_MODEL = "claude-haiku-4-5"
LARGE_MODEL = "claude-opus-5"

# A stand-in for the thing every RAG/agent service re-sends on every request:
# a long, stable system prompt.
#
# The length is load-bearing. Every model has a MINIMUM CACHEABLE PREFIX, and a
# prompt below it is silently not cached -- no error, just
# cache_creation_input_tokens: 0. The minimum is not monotonic across
# generations: it is 512 tokens on Claude Opus 5 but 4096 on Claude Haiku 4.5,
# so a prompt that caches fine on the big model can silently fail to cache on
# the small one. That asymmetry is a real production trap for exactly the
# two-tier routed architecture this talk recommends. We size well past 4096 so
# both tiers actually cache, and assert it at record time rather than trusting it.
SYSTEM_PROMPT = (
    "You are a support assistant for an infrastructure company.\n\n"
    + "\n".join(
        f"Policy {i}: When a customer asks about topic {i}, first check their "
        f"account tier, then consult the tier-{i % 4} runbook, then answer in at "
        f"most three sentences. Never promise a refund without an approval code. "
        f"Escalate to a human if the customer mentions a legal or compliance issue."
        for i in range(1, 120)
    )
)

QUERIES = [
    "How do I reset my password?",
    "What is your refund policy for annual plans?",
    "Explain the trade-offs between our multi-region failover designs and recommend one.",
    "Summarize this month's incident trends.",
]


@dataclass
class CallRecord:
    label: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        p = PRICES.get(self.model, {"in": 1.0, "out": 5.0})
        return (
            self.input_tokens * p["in"]
            + self.cache_creation_input_tokens * p["in"] * CACHE_WRITE_MULT
            + self.cache_read_input_tokens * p["in"] * CACHE_READ_MULT
            + self.output_tokens * p["out"]
        ) / 1e6

    @property
    def prompt_tokens_total(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


@dataclass
class Cassette:
    recorded_at: str = ""
    calls: list[dict] = field(default_factory=list)
    note: str = ""

    def records(self) -> list[CallRecord]:
        return [CallRecord(**c) for c in self.calls]


def _stream_call(client, model: str, query: str, cached: bool, label: str) -> CallRecord:
    """One streamed call. Streaming is how we get a real TTFT rather than a
    round-trip time that hides where the latency actually went."""
    system = [{"type": "text", "text": SYSTEM_PROMPT}]
    if cached:
        system[0]["cache_control"] = {"type": "ephemeral"}

    t0 = time.perf_counter()
    ttft = 0.0
    with client.messages.stream(
        model=model,
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": query}],
    ) as stream:
        for _ in stream.text_stream:
            if ttft == 0.0:
                ttft = (time.perf_counter() - t0) * 1000.0
        msg = stream.get_final_message()
    total = (time.perf_counter() - t0) * 1000.0
    u = msg.usage
    return CallRecord(
        label=label,
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        ttft_ms=ttft,
        total_ms=total,
    )


def record() -> Cassette:
    """Make the real calls. Deliberately small: 150 max_tokens, 8 calls."""
    import anthropic

    client = anthropic.Anthropic()
    calls: list[CallRecord] = []

    # -- prefix caching: cold write, then two warm reads -------------------
    calls.append(_stream_call(client, SMALL_MODEL, QUERIES[0], True, "cache-cold"))
    time.sleep(1.0)  # the entry becomes readable once the first response starts
    calls.append(_stream_call(client, SMALL_MODEL, QUERIES[1], True, "cache-warm-1"))
    calls.append(_stream_call(client, SMALL_MODEL, QUERIES[3], True, "cache-warm-2"))
    # and the control: identical prompt, caching off
    calls.append(_stream_call(client, SMALL_MODEL, QUERIES[1], False, "cache-off"))

    # -- routing: same work, two tiers ------------------------------------
    for q, tag in ((QUERIES[0], "easy"), (QUERIES[2], "hard")):
        calls.append(_stream_call(client, SMALL_MODEL, q, False, f"route-small-{tag}"))
        calls.append(_stream_call(client, LARGE_MODEL, q, False, f"route-large-{tag}"))

    cold = calls[0]
    if cold.cache_creation_input_tokens == 0:
        raise RuntimeError(
            f"Nothing was cached: the {cold.prompt_tokens_total}-token prefix is "
            f"below {cold.model}'s minimum cacheable prefix. Lengthen SYSTEM_PROMPT."
        )

    cas = Cassette(
        recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        calls=[asdict(c) for c in calls],
        note="Real Anthropic API measurements. Prices are published list rates.",
    )
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
    CASSETTE.write_text(json.dumps(asdict(cas), indent=2))
    return cas


def load() -> Cassette | None:
    if not CASSETTE.exists():
        return None
    return Cassette(**json.loads(CASSETTE.read_text()))


def summarize(cas: Cassette) -> dict:
    """Turn the raw call records into the two claims we wanted to check."""
    by = {c.label: c for c in cas.records()}
    out: dict = {"recorded_at": cas.recorded_at}

    cold, warm, off = by.get("cache-cold"), by.get("cache-warm-1"), by.get("cache-off")
    if cold and warm and off:
        out["cache"] = {
            "prefix_tokens": cold.prompt_tokens_total,
            "cold_cost": cold.cost_usd,
            "warm_cost": warm.cost_usd,
            "uncached_cost": off.cost_usd,
            "tokens_served_from_cache": warm.cache_read_input_tokens,
            "cache_hit_rate_pct": (
                100.0 * warm.cache_read_input_tokens / max(warm.prompt_tokens_total, 1)
            ),
            "cost_reduction_pct": (
                100.0 * (1 - warm.cost_usd / off.cost_usd) if off.cost_usd else 0.0
            ),
            "ttft_uncached_ms": off.ttft_ms,
            "ttft_cached_ms": warm.ttft_ms,
        }

    small, large = by.get("route-small-easy"), by.get("route-large-easy")
    if small and large:
        out["routing"] = {
            "small_model": small.model,
            "large_model": large.model,
            "small_cost": small.cost_usd,
            "large_cost": large.cost_usd,
            "cost_ratio": large.cost_usd / small.cost_usd if small.cost_usd else 0.0,
            "small_ttft_ms": small.ttft_ms,
            "large_ttft_ms": large.ttft_ms,
        }
    return out


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "record":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            print("Set ANTHROPIC_API_KEY (or run `ant auth login`) first.")
            raise SystemExit(1)
        print("Recording ~8 short calls (max_tokens=150). Expect a few cents.")
        print(json.dumps(summarize(record()), indent=2))
    else:
        cas = load()
        if cas is None:
            print(f"No cassette at {CASSETTE}. Run: python -m infeng.realbench record")
            raise SystemExit(1)
        print(json.dumps(summarize(cas), indent=2))
