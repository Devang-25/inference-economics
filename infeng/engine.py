"""Iteration-level inference server simulation.

Models what a real serving engine actually does, one scheduler iteration at a
time: admit what fits, run either a prefill batch or a decode batch, charge the
clock the roofline cost of that batch, retire finished sequences. Latency is
therefore an OUTPUT of queueing and contention, not a parameter -- which is the
only way a cost model tells you anything you didn't already assume.

Two batching policies, because the gap between them is the single largest
cheap win available to most teams:

  STATIC     -- form a batch, prefill it, decode until the LONGEST sequence in
                the batch finishes, then release the whole batch and form the
                next one. Every sequence that finished early leaves its slot
                idle for the rest of the batch, and every request that arrives
                mid-batch waits for the whole thing to drain. Both the GPU and
                the caller pay for the longest member of the batch.

  CONTINUOUS -- schedule at ITERATION granularity. A sequence that finishes is
                retired immediately and its slot is refilled from the waiting
                queue on the very next step. No slot goes idle behind a long
                generation, and TTFT stops being hostage to an unrelated
                request's output length.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .kvcache import ContiguousAllocator, OutOfKVMemory, PagedAllocator
from .metrics import SLO, ReplicaResult
from .prefixcache import RadixPrefixCache
from .quant import FP16, Quantization
from .responsecache import ResponseCache
from .roofline import Executor
from .speculative import SpecConfig, SpeculativeDecoder
from .trace import Request


@dataclass
class ServerConfig:
    """One serving configuration. The demo is a walk through seven of these."""
    name: str
    batching: str = "continuous"          # "static" | "continuous"
    allocator: str = "paged"              # "contiguous" | "paged"
    static_batch_size: int = 8
    max_num_seqs: int = 256
    max_batched_tokens: int = 8192        # prefill token budget per iteration
    chunked_prefill: bool = True
    block_size: int = 16
    max_model_len: int = 32768            # advertised context window
    quant: Quantization = FP16
    prefix_cache: bool = True
    response_cache: bool = False
    spec: SpecConfig | None = None
    max_batch_wait_s: float = 0.5         # static batching: how long to wait to fill a batch
    admission_control: bool = False
    max_queue_depth: int = 4096
    queue_drop_after_s: float = 30.0


@dataclass
class _Seq:
    req: Request
    prefill_remaining: int
    cached_prefix_tokens: int
    shared_blocks: list[int]
    generated: float = 0.0
    kv_len: int = 0
    first_token_s: float = -1.0
    finish_s: float = -1.0
    alpha: float = 0.0
    _kv_debt: float = 0.0


def simulate_replica(
    cfg: ServerConfig,
    requests: list[Request],
    executor: Executor,
    slo: SLO,
    draft: Executor | None = None,
) -> ReplicaResult:
    """Run one replica over one shard of the trace. Deterministic."""
    res = ReplicaResult()
    if not requests:
        return res

    reqs = sorted(requests, key=lambda r: r.arrival_s)

    # -- KV pool ----------------------------------------------------------
    capacity = executor.max_kv_tokens
    if cfg.allocator == "paged":
        alloc = PagedAllocator(capacity, cfg.block_size, cfg.max_model_len)
    else:
        # Static batching rebuilds the batch tensor each time, so it only needs
        # the batch's longest member. Continuous batching cannot, so it pays the
        # full context window per slot.
        alloc = ContiguousAllocator(
            capacity, cfg.max_model_len,
            reserve_mode="dynamic" if cfg.batching == "static" else "model_max",
        )

    prefix = (
        RadixPrefixCache(cfg.block_size, max_blocks=int(alloc_blocks(alloc) * 0.35))
        if (cfg.prefix_cache and cfg.allocator == "paged")
        else None
    )
    rcache = ResponseCache() if cfg.response_cache else None
    spec = (
        SpeculativeDecoder(executor, draft, cfg.spec)
        if (cfg.spec and cfg.spec.enabled and draft is not None)
        else None
    )

    waiting: deque[_Seq] = deque()
    running: list[_Seq] = []
    t = 0.0
    idx = 0
    n = len(reqs)
    spec_emitted_total = 0.0
    spec_steps = 0
    spec_alpha_sum = 0.0

    def admit_arrivals(now: float) -> None:
        nonlocal idx
        while idx < n and reqs[idx].arrival_s <= now:
            r = reqs[idx]
            idx += 1
            res.prompt_tokens += r.prompt_tokens

            # Tier 0: never touch the GPU if we already know the answer.
            if rcache is not None:
                hit = rcache.get(r.prompt_text)
                if hit is not None:
                    res.requests_cached += 1
                    res.requests_completed += 1
                    res.output_tokens += r.output_tokens
                    res.ttft_ms.append(4.0)     # cache lookup + network
                    res.tpot_ms.append(0.0)
                    res.e2e_ms.append(6.0)
                    continue
                rcache.put(r.prompt_text, r.prompt_tokens, r.output_tokens)

            if cfg.admission_control and len(waiting) >= cfg.max_queue_depth:
                res.requests_dropped += 1
                continue
            waiting.append(_Seq(
                req=r, prefill_remaining=r.prompt_tokens,
                cached_prefix_tokens=0, shared_blocks=[],
                alpha=cfg.spec.acceptance_for(r.difficulty) if cfg.spec else 0.0,
            ))

    def try_admit_to_kv(s: _Seq) -> bool:
        """Prefix-cache lookup then KV reservation. Returns False as
        backpressure, which is a scheduling signal, not an error."""
        shared, hit = ([], 0)
        if prefix is not None:
            shared, hit = prefix.lookup(s.req.token_ids)
        try:
            if cfg.allocator == "paged":
                if not alloc.can_admit(s.req.prompt_tokens, shared):
                    freed = prefix.evict_to_fit(8) if prefix else []
                    for b in freed:
                        alloc.unpin([b])
                    if not alloc.can_admit(s.req.prompt_tokens, shared):
                        return False
                alloc.admit(s.req.rid, s.req.prompt_tokens, shared)
            else:
                if not alloc.can_admit(s.req.prompt_tokens):
                    return False
                alloc.admit(s.req.rid, s.req.prompt_tokens)
        except OutOfKVMemory:
            return False
        s.shared_blocks = shared
        s.cached_prefix_tokens = hit
        s.prefill_remaining = s.req.prompt_tokens - hit
        s.kv_len = s.req.prompt_tokens
        res.prefill_tokens_skipped += hit
        return True

    def retire(s: _Seq, now: float) -> None:
        res.requests_completed += 1
        res.output_tokens += s.req.output_tokens
        res.e2e_ms.append((now - s.req.arrival_s) * 1000.0)
        if s.req.output_tokens > 1 and s.first_token_s >= 0:
            res.tpot_ms.append(
                (now - s.first_token_s) * 1000.0 / (s.req.output_tokens - 1)
            )
        keep = []
        if prefix is not None:
            blocks = alloc.blocks_of(s.req.rid)
            n_prompt_blocks = s.req.prompt_tokens // cfg.block_size
            keep = prefix.insert(s.req.token_ids, blocks[:n_prompt_blocks])
            alloc.pin(keep)
        alloc.release(s.req.rid, keep_blocks=keep)

    # ---------------------------------------------------------------- static
    if cfg.batching == "static":
        while idx < n or waiting or running:
            admit_arrivals(t)
            if not waiting and not running:
                if idx < n:
                    nxt = reqs[idx].arrival_s
                    res.idle_s += max(0.0, nxt - t)
                    t = nxt
                    continue
                break

            # A real queue-and-batch server WAITS to fill the batch rather than
            # running whatever happens to be queued -- running batches of 1 on a
            # memory-bound decode would be indefensibly wasteful, and modelling
            # it that way would strawman the baseline. Wait for a full batch, or
            # until the oldest waiter times out.
            if (waiting and len(waiting) < cfg.static_batch_size
                    and idx < n
                    and (t - waiting[0].req.arrival_s) < cfg.max_batch_wait_s):
                nxt = reqs[idx].arrival_s
                deadline = waiting[0].req.arrival_s + cfg.max_batch_wait_s
                step_to = min(nxt, deadline)
                if step_to > t:
                    res.idle_s += step_to - t
                    t = step_to
                    continue

            batch: list[_Seq] = []
            while waiting and len(batch) < cfg.static_batch_size:
                s = waiting[0]
                if not try_admit_to_kv(s):
                    break
                batch.append(s)
                waiting.popleft()
            if not batch:
                # KV is full and nothing is running: only possible if a single
                # request cannot fit. Drop it rather than deadlock.
                if not running and waiting:
                    waiting.popleft()
                    res.requests_dropped += 1
                continue

            # one prefill pass for the whole batch
            ptoks = sum(s.prefill_remaining for s in batch)
            avg_ctx = sum(s.req.prompt_tokens for s in batch) / len(batch)
            dt = executor.prefill_time_s(ptoks, avg_ctx)
            t += dt
            res.busy_s += dt
            res.prefill_s += dt
            res.prefill_tokens_computed += ptoks
            for s in batch:
                s.first_token_s = t
                s.generated = 1.0
                res.ttft_ms.append((t - s.req.arrival_s) * 1000.0)

            # decode until the LONGEST sequence in the batch is done -- the
            # defining cost of static batching
            steps = max(s.req.output_tokens for s in batch) - 1
            for _ in range(max(steps, 0)):
                live = [s for s in batch if s.generated < s.req.output_tokens]
                if not live:
                    break
                # the batch slot is held for the whole run, so the engine still
                # pays for every slot in the batch, finished or not
                kv = sum(s.kv_len for s in batch)
                dt = executor.decode_step_time_s(len(batch), kv)
                t += dt
                res.busy_s += dt
                res.decode_s += dt
                res.batch_sizes.append(len(live))
                for s in live:
                    s.generated += 1
                    s.kv_len += 1
                    alloc.append(s.req.rid, 1)
            for s in batch:
                retire(s, t)
            res.kv_utilization.append(
                100.0 * alloc.stats.reserved_tokens / max(alloc.capacity, 1)
            )
            res.kv_waste_samples.append(alloc.waste_now_pct())
            res.kv_residency_samples.append(alloc.cache_residency_pct())

    # ------------------------------------------------------------ continuous
    else:
        while idx < n or waiting or running:
            admit_arrivals(t)
            if not waiting and not running:
                if idx < n:
                    nxt = reqs[idx].arrival_s
                    res.idle_s += max(0.0, nxt - t)
                    t = nxt
                    continue
                break

            # --- schedule a prefill batch if anything is waiting -----------
            prefill_batch: list[_Seq] = []
            budget = cfg.max_batched_tokens
            while waiting and len(running) + len(prefill_batch) < cfg.max_num_seqs:
                s = waiting[0]
                if not try_admit_to_kv(s):
                    break                      # KV backpressure
                need = s.prefill_remaining
                if need > budget:
                    if cfg.chunked_prefill and not prefill_batch:
                        # split the giant prompt so it cannot stall every
                        # decode in flight behind it
                        pass
                    elif prefill_batch:
                        alloc.release(s.req.rid)
                        break
                prefill_batch.append(s)
                waiting.popleft()
                budget -= min(need, budget)
                if budget <= 0:
                    break

            if prefill_batch:
                if cfg.chunked_prefill:
                    ptoks = min(sum(s.prefill_remaining for s in prefill_batch),
                                cfg.max_batched_tokens)
                else:
                    ptoks = sum(s.prefill_remaining for s in prefill_batch)
                avg_ctx = sum(s.req.prompt_tokens for s in prefill_batch) / len(prefill_batch)
                dt = executor.prefill_time_s(ptoks, avg_ctx)
                t += dt
                res.busy_s += dt
                res.prefill_s += dt
                res.prefill_tokens_computed += ptoks
                for s in prefill_batch:
                    s.first_token_s = t
                    s.generated = 1.0
                    res.ttft_ms.append((t - s.req.arrival_s) * 1000.0)
                    running.append(s)
                continue

            if not running:
                continue

            # --- decode iteration ------------------------------------------
            bs = len(running)
            kv = alloc.live_kv_tokens()
            if spec is not None:
                mean_alpha = sum(s.alpha for s in running) / bs
                dt, emitted = spec.step_time_and_tokens(bs, kv, mean_alpha)
                spec_emitted_total += emitted
                spec_alpha_sum += mean_alpha
                spec_steps += 1
            else:
                dt = executor.decode_step_time_s(bs, kv)
                emitted = 1.0
            t += dt
            res.busy_s += dt
            res.decode_s += dt
            res.batch_sizes.append(bs)

            done: list[_Seq] = []
            for s in running:
                s.generated += emitted
                s._kv_debt += emitted
                whole = int(s._kv_debt)
                if whole:
                    s._kv_debt -= whole
                    s.kv_len += whole
                    if cfg.allocator == "paged" and not alloc.has_room_to_grow(s.req.rid, whole):
                        freed = prefix.evict_to_fit(4) if prefix else []
                        for b in freed:
                            alloc.unpin([b])
                    try:
                        alloc.append(s.req.rid, whole)
                    except OutOfKVMemory:
                        alloc.preempt(s.req.rid)
                        res.preemptions += 1
                        s.generated = 0.0
                        s.kv_len = s.req.prompt_tokens
                        waiting.appendleft(s)
                        done.append(s)
                        continue
                if s.generated >= s.req.output_tokens:
                    done.append(s)
                    retire(s, t)
            if done:
                dset = {id(x) for x in done}
                running = [s for s in running if id(s) not in dset]
            res.kv_utilization.append(
                100.0 * alloc.stats.reserved_tokens / max(alloc.capacity, 1)
            )
            res.kv_waste_samples.append(alloc.waste_now_pct())
            res.kv_residency_samples.append(alloc.cache_residency_pct())

    res.makespan_s = t
    res.kv_waste_pct = (
        sum(res.kv_waste_samples) / len(res.kv_waste_samples)
        if res.kv_waste_samples else 0.0
    )
    res.cache_residency_pct = (
        sum(res.kv_residency_samples) / len(res.kv_residency_samples)
        if res.kv_residency_samples else 0.0
    )
    res.preemptions = max(res.preemptions, alloc.stats.preemptions)
    if prefix is not None:
        res.prefix_hit_pct = prefix.stats.token_hit_rate_pct
    if rcache is not None:
        res.cache_hit_pct = rcache.stats.hit_rate_pct
    if spec_steps:
        res.spec_accept_pct = 100.0 * spec_alpha_sum / spec_steps
        res.spec_tokens_per_step = spec_emitted_total / spec_steps
    return res


def alloc_blocks(alloc) -> int:
    return getattr(alloc, "n_blocks", alloc.capacity // 16)
