"""Roofline execution-time model for a tensor-parallel transformer server.

Two regimes, and conflating them is the root of most bad inference cost models:

  PREFILL  -- one forward pass over many tokens at once. Large GEMMs, high
              arithmetic intensity => COMPUTE bound. Cost scales with prompt
              tokens. This sets TTFT.

  DECODE   -- one forward pass per token per sequence. The GEMMs are tall and
              skinny, so the step is dominated by streaming the whole weight
              matrix out of HBM => MEMORY-BANDWIDTH bound. Cost per STEP is
              nearly independent of batch size, which is the entire economic
              argument for batching: the weight read is amortized across every
              sequence in the batch. This sets TPOT.

Every stage in the demo is an attempt to move one of these two lines.
"""
from __future__ import annotations

from dataclasses import dataclass

from .hardware import GPU
from .model import Model
from .quant import FP16, Quantization


@dataclass
class Efficiency:
    """Achieved fractions of peak. These are the only fitted numbers in the
    model; every other input is a published spec or derived from architecture.
    Calibrated in bench/calibration.json against public benchmark points."""
    mfu_prefill: float = 0.45        # FlashAttention-class prefill on H100
    mfu_decode: float = 0.65         # tall-skinny GEMMs use tensor cores poorly
    bandwidth_efficiency: float = 0.80
    step_overhead_s: float = 0.0015  # kernel launch + sample + detokenize + sched
    tp_efficiency: float = 0.85      # all-reduce is not free and does not overlap fully


@dataclass
class Executor:
    """Binds a model to a physical deployment (GPU type, TP degree, quantization)
    and answers: how long does this batch take?"""
    gpu: GPU
    model: Model
    quant: Quantization = FP16
    tp: int = 4
    eff: Efficiency = None

    def __post_init__(self):
        if self.eff is None:
            self.eff = Efficiency()

    # -- capacity ---------------------------------------------------------
    @property
    def weight_bytes_per_gpu(self) -> float:
        return self.model.weight_bytes(self.quant) / self.tp

    @property
    def kv_bytes_per_token_per_gpu(self) -> float:
        return self.model.kv_bytes_per_token(self.quant) / self.tp

    @property
    def activation_reserve_bytes(self) -> float:
        """Workspace CUDA graphs, activations and fragmentation headroom.
        Real servers reserve ~10% of the card; pretending otherwise inflates
        every KV-capacity number you publish."""
        return 0.10 * self.gpu.memory_bytes

    @property
    def kv_budget_bytes_per_gpu(self) -> float:
        free = self.gpu.memory_bytes - self.weight_bytes_per_gpu - self.activation_reserve_bytes
        return max(free, 0.0)

    @property
    def max_kv_tokens(self) -> int:
        """Hard ceiling on total cached tokens across all live sequences.
        This is the number PagedAttention exists to stop you from wasting."""
        return int(self.kv_budget_bytes_per_gpu / self.kv_bytes_per_token_per_gpu)

    # -- timing -----------------------------------------------------------
    def _tp_allreduce_s(self, n_tokens: int) -> float:
        if self.tp == 1:
            return 0.0
        # 2 all-reduces per layer (attn out, ffn out), ring cost factor 2(N-1)/N
        payload = n_tokens * self.model.hidden * 2  # bf16 activations
        factor = 2.0 * (self.tp - 1) / self.tp
        per_layer = 2 * payload * factor / self.gpu.nvlink_bps
        return self.model.layers * per_layer

    def prefill_time_s(self, n_tokens: int, avg_context: float) -> float:
        """Time to prefill `n_tokens` of prompt where sequences average
        `avg_context` in length (the attention term is context-dependent)."""
        if n_tokens <= 0:
            return 0.0
        flops = (
            self.model.flops_per_token * n_tokens * self.quant.prefill_overhead
            + self.model.attention_flops(int(avg_context), n_tokens)
        )
        peak = self.gpu.flops_for(self.quant.compute_dtype)
        compute_s = (flops / self.tp) / (peak * self.eff.mfu_prefill)
        # weights must still be streamed once for the pass
        mem_s = self.weight_bytes_per_gpu / (
            self.gpu.hbm_bandwidth_bps * self.eff.bandwidth_efficiency
        )
        core = max(compute_s, mem_s) / self.eff.tp_efficiency
        return core + self._tp_allreduce_s(n_tokens) + self.eff.step_overhead_s

    def decode_step_time_s(self, batch_size: int, kv_tokens_in_batch: float) -> float:
        """Time for ONE decode iteration producing `batch_size` tokens.

        Note the shape of this function: the dominant term (weight bytes) is
        constant in batch_size. Doubling the batch roughly doubles throughput
        at near-constant latency -- until the KV term or compute takes over.
        """
        if batch_size <= 0:
            return 0.0
        mem_bytes = (
            self.weight_bytes_per_gpu
            + kv_tokens_in_batch * self.kv_bytes_per_token_per_gpu
        )
        mem_s = mem_bytes / (self.gpu.hbm_bandwidth_bps * self.eff.bandwidth_efficiency)
        peak = self.gpu.flops_for(self.quant.compute_dtype)
        compute_s = (self.model.flops_per_token * batch_size / self.tp) / (
            peak * self.eff.mfu_decode
        )
        core = max(mem_s, compute_s) / self.eff.tp_efficiency
        return core + self._tp_allreduce_s(batch_size) + self.eff.step_overhead_s

    # -- economics --------------------------------------------------------
    @property
    def usd_per_hour(self) -> float:
        return self.tp * self.gpu.usd_per_hour

    def usd_per_million_tokens(self, tokens_per_second: float) -> float:
        if tokens_per_second <= 0:
            return float("inf")
        return self.usd_per_hour / (tokens_per_second * 3600.0) * 1e6
