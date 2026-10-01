"""Latency/throughput/cost accounting.

Percentiles, not means. A mean TTFT of 400ms hides the fact that one request in
twenty waited four seconds, and it is the p95 that your users churn over and
your SLO is written against.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class SLO:
    """The contract. Every optimization is scored as: how few GPUs can hold
    this line?"""
    ttft_p95_ms: float = 1000.0
    tpot_p95_ms: float = 50.0
    name: str = "interactive"

    def holds(self, ttft_p95_ms: float, tpot_p95_ms: float) -> bool:
        return ttft_p95_ms <= self.ttft_p95_ms and tpot_p95_ms <= self.tpot_p95_ms


@dataclass
class ReplicaResult:
    """Everything one replica measured over the trace."""
    requests_completed: int = 0
    requests_cached: int = 0
    requests_dropped: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefill_tokens_computed: int = 0     # after prefix-cache savings
    prefill_tokens_skipped: int = 0
    busy_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    idle_s: float = 0.0
    makespan_s: float = 0.0
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    e2e_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    kv_utilization: list[float] = field(default_factory=list)
    kv_waste_samples: list[float] = field(default_factory=list)
    kv_residency_samples: list[float] = field(default_factory=list)
    cache_residency_pct: float = 0.0
    preemptions: int = 0
    kv_waste_pct: float = 0.0
    prefix_hit_pct: float = 0.0
    cache_hit_pct: float = 0.0
    spec_accept_pct: float = 0.0
    spec_tokens_per_step: float = 0.0

    # -- derived ----------------------------------------------------------
    @property
    def ttft_p50(self) -> float: return percentile(self.ttft_ms, 50)
    @property
    def ttft_p95(self) -> float: return percentile(self.ttft_ms, 95)
    @property
    def ttft_p99(self) -> float: return percentile(self.ttft_ms, 99)
    @property
    def tpot_p50(self) -> float: return percentile(self.tpot_ms, 50)
    @property
    def tpot_p95(self) -> float: return percentile(self.tpot_ms, 95)
    @property
    def e2e_p95(self) -> float: return percentile(self.e2e_ms, 95)

    @property
    def avg_batch(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def avg_kv_util(self) -> float:
        return sum(self.kv_utilization) / len(self.kv_utilization) if self.kv_utilization else 0.0

    @property
    def gpu_utilization_pct(self) -> float:
        return 100.0 * self.busy_s / self.makespan_s if self.makespan_s else 0.0

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.makespan_s if self.makespan_s else 0.0

    @property
    def total_tokens_per_s(self) -> float:
        return (self.prompt_tokens + self.output_tokens) / self.makespan_s if self.makespan_s else 0.0


@dataclass
class FleetResult:
    """A replica result projected onto a fleet, plus the economics."""
    stage: str
    replicas: int
    gpus_per_replica: int
    usd_per_gpu_hour: float
    replica: ReplicaResult
    slo: SLO
    slo_held: bool
    quality_delta_pct: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def gpus(self) -> int:
        return self.replicas * self.gpus_per_replica

    @property
    def fleet_usd_per_hour(self) -> float:
        return self.gpus * self.usd_per_gpu_hour

    @property
    def billable_tokens_per_s(self) -> float:
        """Tokens the CUSTOMER is billed for, fleet-wide -- including tokens we
        served from cache and never computed. Serving a cached answer still
        earns revenue; that is exactly why caching moves the cost line."""
        return self.detail.get("billable_tokens_per_s", 0.0)

    @property
    def usd_per_million_tokens(self) -> float:
        tps = self.billable_tokens_per_s
        if tps <= 0:
            return float("inf")
        return self.fleet_usd_per_hour / (tps * 3600.0) * 1e6

    @property
    def usd_per_million_output_tokens(self) -> float:
        tps = self.detail.get("billable_output_tokens_per_s", 0.0)
        if tps <= 0:
            return float("inf")
        return self.fleet_usd_per_hour / (tps * 3600.0) * 1e6

    @property
    def monthly_usd(self) -> float:
        return self.fleet_usd_per_hour * 24 * 30
