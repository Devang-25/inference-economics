"""GPU hardware profiles.

Numbers are vendor-published peak specs. Achievable fractions (MFU for compute,
bandwidth efficiency for memory) are separate, calibrated knobs -- see
``bench/calibration.json``. Keeping peak specs and achieved efficiency apart is
deliberate: peaks are facts, efficiencies are measurements we must justify.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    memory_bytes: int           # HBM capacity
    hbm_bandwidth_bps: float    # peak HBM bandwidth, bytes/sec
    bf16_flops: float           # peak dense BF16/FP16 tensor-core FLOP/s
    fp8_flops: float            # peak dense FP8 FLOP/s
    int8_flops: float           # peak dense INT8 OP/s
    nvlink_bps: float           # per-GPU bidirectional NVLink bandwidth, bytes/sec
    usd_per_hour: float         # on-demand street price per GPU-hour

    def flops_for(self, compute_dtype: str) -> float:
        return {
            "fp16": self.bf16_flops,
            "bf16": self.bf16_flops,
            "fp8": self.fp8_flops,
            "int8": self.int8_flops,
        }[compute_dtype]


# H100 SXM5 80GB: 989 TFLOP/s dense BF16, 1979 dense FP8/INT8, 3.35 TB/s HBM3.
H100_SXM = GPU(
    name="H100-SXM-80GB",
    memory_bytes=80 * 1024**3,
    hbm_bandwidth_bps=3.35e12,
    bf16_flops=989e12,
    fp8_flops=1979e12,
    int8_flops=1979e12,
    nvlink_bps=450e9,
    usd_per_hour=2.50,
)

# A100 SXM4 80GB: 312 TFLOP/s dense BF16, 624 INT8, 2.039 TB/s HBM2e.
# Kept because the PagedAttention/vLLM paper numbers we validate against are A100.
A100_SXM = GPU(
    name="A100-SXM-80GB",
    memory_bytes=80 * 1024**3,
    hbm_bandwidth_bps=2.039e12,
    bf16_flops=312e12,
    fp8_flops=624e12,   # A100 has no FP8; mirrors INT8 so the model stays total
    int8_flops=624e12,
    nvlink_bps=300e9,
    usd_per_hour=1.80,
)

L40S = GPU(
    name="L40S-48GB",
    memory_bytes=48 * 1024**3,
    hbm_bandwidth_bps=864e9,
    bf16_flops=362e12,
    fp8_flops=733e12,
    int8_flops=733e12,
    nvlink_bps=64e9,    # PCIe Gen4 x16, no NVLink
    usd_per_hour=1.10,
)

CATALOG = {g.name: g for g in (H100_SXM, A100_SXM, L40S)}
