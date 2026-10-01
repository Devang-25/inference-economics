"""The seven stages, and how a stage is turned into a bill.

The central question this module answers is NOT "how fast is it." It's:

    Given this traffic and this latency SLO, how many GPUs do I need?

That reframing is the whole point. Throughput numbers are easy to cherry-pick
by relaxing latency; fleet size at a fixed SLO is not. Every stage is sized by
binary search over replica count until p95 TTFT and p95 TPOT both hold, and the
dollar figure falls out of the fleet that survived.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .engine import ServerConfig, simulate_replica
from .hardware import GPU, H100_SXM
from .metrics import SLO, FleetResult, ReplicaResult
from .model import LLAMA_31_8B, LLAMA_31_70B, LLAMA_32_1B, Model
from .quant import AWQ_INT4, FP16, FP8, INT8, Quantization
from .roofline import Efficiency, Executor
from .router import DifficultyRouter
from .speculative import SpecConfig
from .trace import Request, Workload

MAX_REPLICAS = 256

# Tuned on the threshold sweep in bench/router_sweep.py: the knee where the
# small tier takes most of the traffic before under-routing starts to climb.
ROUTER_THRESHOLD = 0.50


@dataclass
class Stage:
    key: str
    title: str
    lever: str                  # the one-line "what changed"
    server: ServerConfig
    routing: bool = False
    tp: int = 4
    model: Model = LLAMA_31_70B
    gpu: GPU = H100_SXM
    draft_model: Model | None = None
    takeaway: str = ""
    # True for the config that represents "stock vLLM defaults" -- the honest
    # starting point for most teams, and the baseline any speedup claim made to
    # a room of practitioners should really be measured against.
    reference: bool = False


def _base(**kw) -> ServerConfig:
    defaults = dict(
        name="cfg", batching="continuous", allocator="paged",
        static_batch_size=32, max_num_seqs=256, max_batched_tokens=8192,
        chunked_prefill=True, block_size=16, max_model_len=32768,
        quant=FP16, prefix_cache=False, response_cache=False, spec=None,
    )
    defaults.update(kw)
    return ServerConfig(**defaults)


STAGES: list[Stage] = [
    Stage(
        key="s0", title="Stock vLLM defaults",
        lever="continuous batching + PagedAttention, FP16",
        server=_base(name="stock", batching="continuous", allocator="paged"),
        reference=True,
        takeaway="The honest starting point. Continuous batching and paged KV "
                 "are already on -- you got those for free by not writing your "
                 "own serving loop. Everything after this is work you must do.",
    ),
    Stage(
        key="s1", title="Prefix caching",
        lever="radix tree over resident KV blocks",
        server=_base(name="prefix", allocator="paged", prefix_cache=True),
        takeaway="System prompts and RAG context get recomputed on every single "
                 "request until you stop doing that. It hits prefill, so it hits "
                 "TTFT and it frees the compute that decode is competing for.",
    ),
    Stage(
        key="s2", title="Quantization (INT8 W8A8)",
        lever="halve weight bytes and KV bytes",
        server=_base(name="int8", allocator="paged", prefix_cache=True, quant=INT8),
        takeaway="Decode is memory-bandwidth bound, so halving the weights nearly "
                 "halves the step. W8A8 halves the KV cache too, which buys back "
                 "concurrency. Note W4A16 would NOT do this -- it leaves compute "
                 "in FP16 and slows prefill down.",
    ),
    Stage(
        key="s3", title="Speculative decoding",
        lever="1B drafts 4 tokens, 70B verifies in one pass",
        server=_base(name="spec", allocator="paged", prefix_cache=True, quant=INT8,
                     spec=SpecConfig(k=4, enabled=True)),
        draft_model=LLAMA_32_1B,
        takeaway="Measured, not assumed -- and at this batch size it buys nothing. "
                 "Speculation is free only while you are memory-bound; by now the "
                 "batch is large enough to be compute-bound, and the extra k+1 "
                 "FLOPs per step compete with the work you already have.",
    ),
    Stage(
        key="s4", title="Routing + response cache",
        lever="most traffic never touches the 70B",
        server=_base(name="routed", allocator="paged", prefix_cache=True, quant=INT8,
                     response_cache=True),
        routing=True,
        takeaway="The biggest remaining lever is architectural, not kernel-level: "
                 "stop sending easy questions to an expensive model, and stop "
                 "answering the same question twice.",
    ),
]


# The pre-vLLM configuration, kept for a matched-fleet comparison rather than as
# the cost anchor. Sizing a fleet for it is not meaningful -- static batching's
# p95 TTFT is floored by batch drain time, so no fleet size reaches an
# interactive SLO -- and using it as the denominator would inflate every number
# in the talk against a strawman nobody in the room is actually running.
LEGACY = Stage(
    key="legacy", title="Static batching (pre-vLLM)",
    lever="form a batch, run it to completion, repeat",
    server=_base(name="legacy", batching="static", allocator="contiguous",
                 static_batch_size=32),
    takeaway="Every slot is held by the longest generation in the batch.",
)


def legacy_comparison(workload: Workload, slo: SLO, replicas: int) -> dict:
    """Static batching vs stock vLLM on the SAME fleet.

    A fair, assumption-free comparison: identical hardware, identical traffic,
    only the scheduler differs. Reports throughput and latency rather than a
    dollar figure, because the honest claim here is "this is what your serving
    framework already bought you", not "look how big my speedup is".
    """
    out = {}
    for key, stage in (("legacy", LEGACY), ("stock", STAGES[0])):
        ex, draft = _executors(stage)
        r = simulate_replica(stage.server, _shard(workload.requests, replicas), ex, slo, draft)
        out[key] = r
    return out


def _shard(requests: list[Request], replicas: int, which: int = 0) -> list[Request]:
    """One replica's share of traffic under a round-robin load balancer.

    Sharding matters for more than arithmetic: a replica only gets prefix-cache
    hits on traffic IT saw, so splitting across more replicas genuinely lowers
    cache locality. Simulating the shard captures that; dividing a global
    number by N would not.
    """
    return [r for i, r in enumerate(requests) if i % replicas == which]


def _executors(stage: Stage, tier: str = "big") -> tuple[Executor, Executor | None]:
    if tier == "big":
        model, tp = stage.model, stage.tp
    else:
        model, tp = LLAMA_31_8B, 1
    ex = Executor(stage.gpu, model, stage.server.quant, tp=tp)
    draft = (
        Executor(stage.gpu, stage.draft_model, stage.server.quant, tp=1)
        if stage.draft_model is not None else None
    )
    return ex, draft


# Replica-count ladder. A linear scan over a geometric ladder rather than a
# binary search, because SLO attainment is NOT monotone in replica count: decode
# is memory-bound, so splitting the same offered load across more replicas
# shrinks each replica's batch and can *reduce* aggregate throughput. Binary
# search silently returns garbage on a non-monotone predicate; the ladder finds
# the smallest fleet that actually holds the line.
REPLICA_LADDER = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 160, 192, 256)


def size_fleet(
    stage: Stage, requests: list[Request], slo: SLO, tier: str = "big",
    max_replicas: int = MAX_REPLICAS, progress=None,
) -> tuple[int, ReplicaResult, bool]:
    """Smallest replica count whose p95 TTFT and p95 TPOT both hold.

    There is deliberately no separate "is it keeping up" test. A fleet that
    cannot keep up builds an unbounded backlog, and an unbounded backlog shows
    up as an unbounded p95 TTFT -- the SLO check already catches it, and adding
    a second criterion only introduced a non-monotone predicate and a bug.

    Returns (replicas, result, slo_held). If no fleet on the ladder holds the
    SLO, returns the largest tried and slo_held=False, because "you cannot buy
    this SLO with more GPUs" is a real answer worth reporting rather than
    hiding behind a bigger number.
    """
    ex, draft = _executors(stage, tier)
    tried: list[tuple[int, ReplicaResult]] = []
    for n in REPLICA_LADDER:
        if n > max_replicas:
            break
        if progress is not None:
            progress(n)
        r = simulate_replica(stage.server, _shard(requests, n), ex, slo, draft)
        tried.append((n, r))
        if r.requests_completed > 0 and slo.holds(r.ttft_p95, r.tpot_p95):
            return n, r, True

    # No fleet on the ladder holds the SLO. Static batching lands here, and it
    # is not an artifact: p95 TTFT is floored by how long the in-flight batch
    # takes to DRAIN -- roughly half the longest generation in it -- and that
    # floor does not move when you add replicas. You cannot buy this SLO.
    #
    # So latency cannot size this fleet, and reporting the ladder ceiling would
    # price the baseline at wherever the loop happened to stop and inflate every
    # downstream speedup. Size it on CAPACITY instead, which is what you would
    # actually do: the smallest fleet that absorbs the offered load at a normal
    # 90% utilization target. It is then reported with slo_held=False, because
    # "cheap enough but misses the SLO at any price" is the real finding.
    horizon = max((r.arrival_s for r in requests), default=0.0)
    return (*_capacity_fleet(tried, horizon), False)


def _capacity_fleet(tried: list[tuple[int, "ReplicaResult"]], horizon: float,
                    target_utilization: float = 0.90):
    """Smallest fleet whose per-replica busy time fits inside the arrival window.

    Work conservation: if a replica needs more GPU-seconds than the traffic
    window is long, it is behind and the backlog grows without bound. Sizing to
    90% utilization is the standard capacity-planning target -- it leaves the
    headroom that burstiness demands.
    """
    usable = [(n, r) for n, r in tried if r.requests_completed > 0]
    if not usable:
        return (tried[-1] if tried else (MAX_REPLICAS, ReplicaResult()))
    for n, r in usable:
        if horizon > 0 and r.busy_s <= target_utilization * horizon:
            return n, r
    return usable[-1]


def _knee(tried: list[tuple[int, "ReplicaResult"]], slack: float = 1.5):
    """Smallest fleet whose p95 TTFT is within `slack` of the best the ladder
    ever achieves.

    Anchoring on the ASYMPTOTE, not on step-to-step improvement: early ladder
    steps are all catastrophic together, so consecutive-gain tests trip
    immediately and pick a fleet that is still hopelessly undersized. Once
    latency has converged to within 50% of its floor, more GPUs are buying
    nothing and that is the fleet a competent team would actually run.
    """
    if not tried:
        return MAX_REPLICAS, ReplicaResult()
    usable = [(n, r) for n, r in tried if r.requests_completed > 0]
    if not usable:
        return tried[-1]
    floor = min(r.ttft_p95 for _, r in usable)
    for n, r in usable:
        if r.ttft_p95 <= floor * slack:
            return n, r
    return usable[-1]


def evaluate(stage: Stage, workload: Workload, slo: SLO) -> FleetResult:
    """Size the fleet for a stage and price it."""
    reqs = workload.requests
    total_tokens = workload.total_prompt_tokens + workload.total_output_tokens
    billable_tps = total_tokens / workload.duration_s
    billable_out_tps = workload.total_output_tokens / workload.duration_s
    quality = stage.server.quant.quality_delta_pct
    detail: dict = {"tiers": {}}

    if not stage.routing:
        n, r, held = size_fleet(stage, reqs, slo)
        gpus = n * stage.tp
        detail.update(
            billable_tokens_per_s=billable_tps,
            billable_output_tokens_per_s=billable_out_tps,
            replicas_big=n, replicas_small=0,
            gpus_big=gpus, gpus_small=0,
        )
        return FleetResult(
            stage=stage.title, replicas=n, gpus_per_replica=stage.tp,
            usd_per_gpu_hour=stage.gpu.usd_per_hour, replica=r, slo=slo,
            slo_held=held, quality_delta_pct=quality, detail=detail,
        )

    # -- two-tier routed fleet -------------------------------------------
    router = DifficultyRouter(threshold=ROUTER_THRESHOLD)
    big_reqs, small_reqs, escalated = [], [], []
    for req in reqs:
        if router.route(req) == "big":
            big_reqs.append(req)
        else:
            small_reqs.append(req)
            # Escalation is not free. A request the small model got wrong and a
            # verifier caught is answered TWICE -- once cheap, once expensive.
            # Charging only the cheap half is how routing demos overstate their
            # savings, so the retry is added to the big tier's real load.
            if req.needs_big_model and router.escalate:
                escalated.append(req)
    big_load = big_reqs + escalated

    n_big, r_big, held_big = (
        size_fleet(stage, big_load, slo, tier="big") if big_load else (0, ReplicaResult(), True)
    )
    small_stage = replace(stage, server=replace(stage.server, spec=None))
    n_small, r_small, held_small = (
        size_fleet(small_stage, small_reqs, slo, tier="small") if small_reqs else (0, ReplicaResult(), True)
    )

    gpus_big, gpus_small = n_big * stage.tp, n_small * 1
    merged = _merge(r_big, r_small)
    quality += router.quality_delta_pct()
    detail.update(
        billable_tokens_per_s=billable_tps,
        billable_output_tokens_per_s=billable_out_tps,
        replicas_big=n_big, replicas_small=n_small,
        gpus_big=gpus_big, gpus_small=gpus_small,
        router=router.stats,
        escalated_requests=len(escalated),
        escalation_pct=100.0 * len(escalated) / max(len(reqs), 1),
        small_share_pct=router.stats.small_share_pct,
        under_route_pct=router.stats.under_route_pct,
        over_route_pct=router.stats.over_route_pct,
    )
    fr = FleetResult(
        stage=stage.title, replicas=n_big + n_small, gpus_per_replica=1,
        usd_per_gpu_hour=stage.gpu.usd_per_hour, replica=merged, slo=slo,
        slo_held=held_big and held_small,
        quality_delta_pct=quality, detail=detail,
    )
    # gpus_per_replica=1 with replicas=total GPUs keeps FleetResult.gpus honest
    fr.replicas = gpus_big + gpus_small
    return fr


def _merge(a: ReplicaResult, b: ReplicaResult) -> ReplicaResult:
    """Blend two tiers' observed latencies into one user-visible distribution."""
    m = ReplicaResult()
    m.requests_completed = a.requests_completed + b.requests_completed
    m.requests_cached = a.requests_cached + b.requests_cached
    m.requests_dropped = a.requests_dropped + b.requests_dropped
    m.prompt_tokens = a.prompt_tokens + b.prompt_tokens
    m.output_tokens = a.output_tokens + b.output_tokens
    m.prefill_tokens_computed = a.prefill_tokens_computed + b.prefill_tokens_computed
    m.prefill_tokens_skipped = a.prefill_tokens_skipped + b.prefill_tokens_skipped
    m.busy_s = a.busy_s + b.busy_s
    m.makespan_s = max(a.makespan_s, b.makespan_s)
    m.ttft_ms = a.ttft_ms + b.ttft_ms
    m.tpot_ms = a.tpot_ms + b.tpot_ms
    m.e2e_ms = a.e2e_ms + b.e2e_ms
    m.batch_sizes = a.batch_sizes + b.batch_sizes
    m.kv_utilization = a.kv_utilization + b.kv_utilization
    m.preemptions = a.preemptions + b.preemptions
    m.kv_waste_pct = max(a.kv_waste_pct, b.kv_waste_pct)
    m.prefix_hit_pct = max(a.prefix_hit_pct, b.prefix_hit_pct)
    m.cache_hit_pct = max(a.cache_hit_pct, b.cache_hit_pct)
    m.spec_accept_pct = a.spec_accept_pct
    m.spec_tokens_per_step = a.spec_tokens_per_step
    return m
