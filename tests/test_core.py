"""Tests for the parts that must be right or every number downstream is wrong."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.validate import run_validation
from infeng.config import STAGES, evaluate
from infeng.hardware import H100_SXM
from infeng.kvcache import ContiguousAllocator, OutOfKVMemory, PagedAllocator
from infeng.metrics import SLO, percentile
from infeng.model import LLAMA_31_8B, LLAMA_31_70B
from infeng.prefixcache import RadixPrefixCache, block_hash
from infeng.quant import AWQ_INT4, FP16, INT8
from infeng.responsecache import ResponseCache
from infeng.roofline import Executor
from infeng.router import DifficultyRouter
from infeng.speculative import expected_accepted
from infeng.trace import generate


# -- architecture math -----------------------------------------------------
def test_param_counts_match_published():
    assert abs(LLAMA_31_70B.total_params / 1e9 - 70.6) < 0.5
    assert abs(LLAMA_31_8B.total_params / 1e9 - 8.03) < 0.1


def test_kv_per_token_is_320KiB_for_70b_fp16():
    assert LLAMA_31_70B.kv_bytes_per_token(FP16) == 320 * 1024


def test_int8_halves_kv_awq_does_not():
    """W8A8 quantizes the KV cache; W4A16 leaves activations (and KV) in FP16.
    Getting this backwards would invent concurrency that does not exist."""
    fp16 = LLAMA_31_70B.kv_bytes_per_token(FP16)
    assert LLAMA_31_70B.kv_bytes_per_token(INT8) == fp16 / 2
    assert LLAMA_31_70B.kv_bytes_per_token(AWQ_INT4) == fp16


# -- roofline --------------------------------------------------------------
def test_decode_step_is_nearly_flat_in_batch_size():
    """The economic argument for batching: the weight read is amortized, so
    16x the batch must cost far less than 16x the time."""
    ex = Executor(H100_SXM, LLAMA_31_70B, FP16, tp=4)
    t1 = ex.decode_step_time_s(1, 2048)
    t16 = ex.decode_step_time_s(16, 16 * 2048)
    assert t16 < 2.0 * t1


def test_awq_speeds_up_decode_but_slows_prefill():
    fp16 = Executor(H100_SXM, LLAMA_31_70B, FP16, tp=4)
    awq = Executor(H100_SXM, LLAMA_31_70B, AWQ_INT4, tp=4)
    assert awq.decode_step_time_s(1, 2048) < fp16.decode_step_time_s(1, 2048)
    assert awq.prefill_time_s(2048, 2048) > fp16.prefill_time_s(2048, 2048)


def test_int8_beats_awq_at_large_batch():
    """W4A16 wins at low batch (memory-bound) and loses at high batch, because
    it never bought any compute. The crossover is the whole point."""
    int8 = Executor(H100_SXM, LLAMA_31_70B, INT8, tp=4)
    awq = Executor(H100_SXM, LLAMA_31_70B, AWQ_INT4, tp=4)
    assert awq.decode_step_time_s(1, 2048) < int8.decode_step_time_s(1, 2048)
    assert awq.decode_step_time_s(512, 512 * 2048) > int8.decode_step_time_s(512, 512 * 2048)


# -- allocators ------------------------------------------------------------
def test_paged_allocator_wastes_only_the_tail_block():
    alloc = PagedAllocator(100_000, block_size=16, max_model_len=32768)
    alloc.admit(1, 1000)
    assert alloc.waste_now_pct() < 4.0


def test_contiguous_model_max_reserves_the_whole_window():
    alloc = ContiguousAllocator(100_000, max_model_len=32768, reserve_mode="model_max")
    alloc.admit(1, 1000)
    assert alloc.waste_now_pct() > 90.0


def test_contiguous_model_max_admits_far_fewer_sequences():
    cap, ctx = 200_000, 32768
    cont = ContiguousAllocator(cap, ctx, reserve_mode="model_max")
    paged = PagedAllocator(cap, 16, ctx)
    n_cont = n_paged = 0
    for i in range(500):
        try:
            cont.admit(i, 900); n_cont += 1
        except OutOfKVMemory:
            break
    for i in range(500):
        try:
            paged.admit(i, 900); n_paged += 1
        except OutOfKVMemory:
            break
    assert n_paged > 5 * n_cont


def test_paged_block_sharing_costs_no_new_memory():
    alloc = PagedAllocator(100_000, 16, 32768)
    alloc.admit(1, 320)
    shared = alloc.blocks_of(1)[:10]
    before = alloc.free_blocks
    alloc.admit(2, 320, shared_blocks=shared)
    assert alloc.free_blocks == before - (20 - 10)


# -- prefix cache ----------------------------------------------------------
def test_block_hash_is_chained_not_content_only():
    """A block must only match if its whole ancestry matches, or the cache
    happily serves you another request's KV."""
    chunk = tuple(range(16))
    assert block_hash(None, chunk) != block_hash(12345, chunk)


def test_prefix_cache_hits_on_shared_system_prompt():
    pc = RadixPrefixCache(block_size=16, max_blocks=1000)
    system = list(range(1, 65))
    a, b = system + list(range(1000, 1032)), system + list(range(2000, 2048))
    pc.lookup(a)
    pc.insert(a, list(range(100, 100 + len(a) // 16)))
    blocks, hit = pc.lookup(b)
    assert hit == 64 and len(blocks) == 4


def test_prefix_cache_never_serves_the_entire_prompt():
    """The model needs at least one token to attend from."""
    pc = RadixPrefixCache(block_size=16, max_blocks=1000)
    ids = list(range(64))
    pc.insert(ids, list(range(10, 14)))
    _, hit = pc.lookup(ids)
    assert hit < len(ids)


# -- response cache --------------------------------------------------------
def test_response_cache_exact_hit_and_unrelated_miss():
    rc = ResponseCache()
    rc.put("How do I reset my password?", 12, 80)
    assert rc.get("how do i reset my password")[0] == "exact"
    assert rc.get("What is the capital of France?") is None


def test_response_cache_declines_paraphrase_at_default_threshold():
    """The conservative threshold is deliberate: a semantic cache that answers
    the wrong question is worse than no cache."""
    rc = ResponseCache()
    rc.put("How do I reset my password?", 12, 80)
    assert rc.get("How can I reset my password?") is None


# -- speculative decoding --------------------------------------------------
def test_expected_accepted_closed_form():
    assert expected_accepted(0.8, 4) == pytest.approx(3.362, abs=1e-3)
    assert expected_accepted(0.0, 4) == pytest.approx(1.0)


# -- router ----------------------------------------------------------------
def test_router_trades_small_share_against_under_routing():
    w = generate(n_requests=600, seed=11)
    lo = DifficultyRouter(threshold=0.30)
    hi = DifficultyRouter(threshold=0.70)
    for r in w.requests:
        lo.route(r); hi.route(r)
    assert hi.stats.small_share_pct > lo.stats.small_share_pct
    assert hi.stats.under_route_pct >= lo.stats.under_route_pct


# -- metrics ---------------------------------------------------------------
def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([], 95) == 0.0


# -- end to end ------------------------------------------------------------
def test_trace_is_deterministic():
    a, b = generate(n_requests=400, seed=7), generate(n_requests=400, seed=7)
    assert [r.prompt_tokens for r in a.requests] == [r.prompt_tokens for r in b.requests]


def test_optimized_stack_is_cheaper_than_stock_and_holds_slo():
    w = generate(n_requests=1200, seed=99)
    slo = SLO(1000, 50)
    ref = evaluate(STAGES[0], w, slo)
    final = evaluate(STAGES[-1], w, slo)
    assert ref.slo_held and final.slo_held
    assert final.gpus < ref.gpus
    assert final.usd_per_million_tokens < ref.usd_per_million_tokens


def test_every_published_reference_point_reproduces():
    for r in run_validation():
        assert abs(r["error_pct"]) <= r["tolerance_pct"], r["name"]
