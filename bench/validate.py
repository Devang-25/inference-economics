"""Validation against published, checkable reference points.

A simulator nobody can check is a slide with extra steps. Every number below
is either (a) exact arithmetic that must reproduce a published figure, or
(b) a published measurement with an explicitly stated tolerance. The tolerance
is part of the claim: a model that reproduces vendor-published architecture
math to within 1% but real serving throughput only to within 30% should say so,
because those are different kinds of confidence.

Where a reference is a range rather than a point (real throughput depends on
kernel versions, batch composition, and tuning), the midpoint is used and the
tolerance is set to cover the published spread.

Run:  python bench/validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infeng.hardware import H100_SXM
from infeng.kvcache import PagedAllocator
from infeng.model import LLAMA_31_70B, LLAMA_31_8B
from infeng.quant import AWQ_INT4, FP16, INT8
from infeng.roofline import Executor
from infeng.speculative import expected_accepted


def _pct_err(modeled: float, reference: float) -> float:
    return 100.0 * (modeled - reference) / reference


def run_validation() -> list[dict]:
    out: list[dict] = []

    def check(name, source, reference_val, modeled_val, tol, fmt="{:,.0f}", unit="",
              kind="point"):
        """kind="point" compares against a published value; kind="upper_bound"
        compares against a published ceiling, where coming in *under* it is a
        pass with zero error -- not an 87% deviation. Scoring a ceiling as a
        point estimate would let a good result inflate the headline error bar,
        which is the opposite of what an error bar is for."""
        if kind == "upper_bound":
            err = 0.0 if modeled_val <= reference_val else _pct_err(modeled_val, reference_val)
        else:
            err = _pct_err(modeled_val, reference_val)
        out.append({
            "name": name,
            "source": source,
            "reference": ("\u2264 " if kind == "upper_bound" else "") + fmt.format(reference_val) + unit,
            "modeled": fmt.format(modeled_val) + unit,
            "error_pct": err,
            "tolerance_pct": tol,
            "kind": kind,
        })

    # -- 1. Architecture math must reproduce published parameter counts -----
    # If these drift, every downstream memory and FLOP number is wrong.
    check("Llama-3.1-70B parameter count", "Meta model card",
          70.6, LLAMA_31_70B.total_params / 1e9, 1.0, "{:.2f}", "B")
    check("Llama-3.1-8B parameter count", "Meta model card",
          8.03, LLAMA_31_8B.total_params / 1e9, 1.0, "{:.2f}", "B")

    # -- 2. KV cache per token: 2 (K,V) x layers x kv_heads x head_dim x dtype
    # 2 * 80 * 8 * 128 * 2 bytes = 327,680 B = 320 KiB. Pure arithmetic.
    check("70B FP16 KV cache per token", "architecture arithmetic",
          320.0, LLAMA_31_70B.kv_bytes_per_token(FP16) / 1024, 1.0, "{:.0f}", " KiB")

    # -- 3. Decode is memory-bandwidth bound -------------------------------
    # At batch 1 the step time is dominated by streaming the weights once:
    #   (131.4 GiB / 4 GPUs) / (3.35 TB/s x 0.80) ~= 13.2 ms
    ex70 = Executor(H100_SXM, LLAMA_31_70B, FP16, tp=4)
    weights_only_ms = 1000 * ex70.weight_bytes_per_gpu / (
        H100_SXM.hbm_bandwidth_bps * ex70.eff.bandwidth_efficiency)
    check("70B TP4 weight-read floor per decode step", "HBM arithmetic",
          13.2, weights_only_ms, 5.0, "{:.1f}", " ms")

    # -- 4. Single-stream decode throughput --------------------------------
    # Community-reported Llama-70B on 4xH100, FP16, batch 1: ~40-70 tok/s.
    step1 = ex70.decode_step_time_s(1, 2048)
    check("70B TP4 FP16 single-stream decode", "published range 40-70 tok/s",
          55.0, 1.0 / step1, 30.0, "{:.0f}", " tok/s")

    # -- 5. Prefill throughput ---------------------------------------------
    # 4xH100 FP16 70B prefill lands around 8-12K tok/s with FlashAttention.
    pf = ex70.prefill_time_s(2048, 2048)
    check("70B TP4 FP16 prefill throughput", "published range 8-12K tok/s",
          10_000.0, 2048 / pf, 30.0, "{:,.0f}", " tok/s")

    # -- 6. INT8 W8A8 decode speedup ---------------------------------------
    # Halving weight bytes in a memory-bound regime should roughly halve the
    # step. Published SmoothQuant / TensorRT-LLM W8A8: 1.5-2.0x on decode.
    ex_int8 = Executor(H100_SXM, LLAMA_31_70B, INT8, tp=4)
    speedup = ex70.decode_step_time_s(1, 2048) / ex_int8.decode_step_time_s(1, 2048)
    check("INT8 W8A8 decode speedup vs FP16", "published range 1.5-2.0x",
          1.75, speedup, 20.0, "{:.2f}", "x")

    # -- 7. W4A16 does NOT speed up prefill --------------------------------
    # AWQ keeps the GEMM in FP16 and adds dequant, so compute-bound prefill
    # gets slightly SLOWER. A model that shows INT4 speeding up prefill is
    # wrong in a way that would mislead a capacity plan.
    ex_awq = Executor(H100_SXM, LLAMA_31_70B, AWQ_INT4, tp=4)
    ratio = ex_awq.prefill_time_s(2048, 2048) / ex70.prefill_time_s(2048, 2048)
    check("AWQ INT4 prefill time vs FP16 (expected >1)", "W4A16 dequant overhead",
          1.12, ratio, 10.0, "{:.2f}", "x")

    # -- 8. PagedAttention internal fragmentation --------------------------
    # The vLLM paper's claim is that paging cuts KV waste to a few percent:
    # only the partial tail block of each sequence is wasted. With 16-token
    # blocks and realistic lengths that is under 4%.
    alloc = PagedAllocator(capacity_tokens=200_000, block_size=16, max_model_len=32768)
    for i, n in enumerate([137, 892, 1503, 64, 2011, 455, 733]):
        alloc.admit(i, n)
    check("PagedAttention KV waste (16-token blocks)", "vLLM paper: <4%",
          4.0, max(alloc.waste_now_pct(), 0.01), 0.0, "{:.2f}", "%", kind="upper_bound")

    # -- 9. Speculative decoding acceptance math ---------------------------
    # E[tokens] = 1 + sum_{i=1..k} alpha^i. For alpha=0.8, k=4 that is 3.362.
    check("Expected tokens/step, alpha=0.8 k=4", "geometric series (closed form)",
          3.362, expected_accepted(0.8, 4), 1.0, "{:.3f}", "")

    # -- 10. KV capacity per replica ---------------------------------------
    # (80 - 32.85 weights - 8 reserve) GiB x 4 GPUs / 320 KiB per token.
    check("70B TP4 FP16 cached tokens per replica", "capacity arithmetic",
          513_000.0, float(ex70.max_kv_tokens), 2.0, "{:,.0f}", "")

    return out


def main() -> int:
    results = run_validation()
    width = max(len(r["name"]) for r in results)
    print(f"{'reference point':<{width}}  {'published':>14}  {'modeled':>14}  "
          f"{'error':>8}  {'tol':>6}  result")
    print("-" * (width + 58))
    failed = 0
    for r in results:
        ok = abs(r["error_pct"]) <= r["tolerance_pct"]
        failed += not ok
        print(f"{r['name']:<{width}}  {r['reference']:>14}  {r['modeled']:>14}  "
              f"{r['error_pct']:>+7.1f}%  {r['tolerance_pct']:>5.0f}%  "
              f"{'PASS' if ok else 'FAIL'}")
    worst = max(abs(r["error_pct"]) for r in results)
    print("-" * (width + 58))
    print(f"{len(results) - failed}/{len(results)} passed · worst-case error {worst:.1f}%")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
