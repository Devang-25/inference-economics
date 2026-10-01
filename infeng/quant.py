"""Quantization schemes and what each one actually buys you.

The distinction that matters in production and that most cost models get wrong:

  W8A8 (INT8 / FP8)  -- weights AND activations quantized. Shrinks the weight
                        bytes read per decode step (memory win) AND doubles
                        tensor-core throughput (compute win). Helps decode and
                        prefill.

  W4A16 (AWQ / GPTQ) -- weights quantized, activations stay FP16. Weights are
                        dequantized on the fly, so the GEMM still runs in FP16.
                        Pure memory win: big speedup for memory-bound DECODE,
                        and a small *penalty* on compute-bound PREFILL from the
                        dequant overhead.

Getting this wrong is how teams ship INT4, see no prefill improvement, and
conclude quantization "doesn't work."

Quality deltas are published averages across standard suites. They are a
starting prior, not a substitute for evaluating on your own traffic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quantization:
    name: str
    weight_bytes_per_param: float  # includes scales/zero-points overhead
    kv_bytes_per_elem: float       # KV cache element width
    compute_dtype: str             # dtype the GEMM actually executes in
    prefill_overhead: float        # multiplier on compute-bound prefill time
    quality_delta_pct: float       # avg accuracy delta vs FP16 baseline, negative = worse
    note: str


FP16 = Quantization(
    name="FP16",
    weight_bytes_per_param=2.0,
    kv_bytes_per_elem=2.0,
    compute_dtype="fp16",
    prefill_overhead=1.0,
    quality_delta_pct=0.0,
    note="baseline",
)

FP8 = Quantization(
    name="FP8 W8A8",
    weight_bytes_per_param=1.0 + 1.0 / 128,   # per-channel scales
    kv_bytes_per_elem=1.0,                    # FP8 KV cache
    compute_dtype="fp8",
    prefill_overhead=1.0,
    quality_delta_pct=-0.1,
    note="Hopper+; halves weights, KV and compute time",
)

INT8 = Quantization(
    name="INT8 W8A8",
    weight_bytes_per_param=1.0 + 2.0 / 128,   # per-channel scales + smoothing
    kv_bytes_per_elem=1.0,
    compute_dtype="int8",
    prefill_overhead=1.0,
    quality_delta_pct=-0.3,
    note="SmoothQuant-style; portable back to Ampere",
)

AWQ_INT4 = Quantization(
    name="AWQ INT4",
    weight_bytes_per_param=0.5 + 2.0 / 128,   # 4-bit + fp16 scale/zero per group of 128
    kv_bytes_per_elem=2.0,                    # activations/KV stay FP16
    compute_dtype="fp16",                     # W4A16: GEMM is still FP16
    prefill_overhead=1.12,                    # dequant cost on compute-bound prefill
    quality_delta_pct=-1.2,
    note="W4A16: big decode win, slight prefill penalty",
)

GPTQ_INT4 = Quantization(
    name="GPTQ INT4",
    weight_bytes_per_param=0.5 + 2.0 / 128,
    kv_bytes_per_elem=2.0,
    compute_dtype="fp16",
    prefill_overhead=1.15,
    quality_delta_pct=-1.8,
    note="W4A16, calibration-sensitive; AWQ usually retains more",
)

CATALOG = {q.name: q for q in (FP16, FP8, INT8, AWQ_INT4, GPTQ_INT4)}
