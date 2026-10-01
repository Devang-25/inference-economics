# Inference Economics

## Author: Devang Sharma

**Engineering LLMs for Cost, Latency, and Scale** — the code behind the
[LLM Day SF 2026](https://llmday.com/2026-san-francisco-q4/) talk.

A fleet-sizing lab for LLM inference. It answers the only question that ends up
on an invoice:

> Given this traffic and this latency SLO, **how many GPUs do I need?**

Throughput numbers are trivially cherry-picked — relax latency and every number
improves. Fleet size at a fixed SLO is not, and it is denominated in the unit
finance actually cares about.

```
$ python demo.py

stage                          GPUs      $/month     TTFT p95     TPOT p95     SLO     vs stock
Stock vLLM defaults              32      $57,600        170ms         40ms    HOLD        1.00x
Prefix caching                   20      $36,000        141ms         42ms    HOLD        1.60x
Quantization (INT8 W8A8)          8      $14,400        111ms         37ms    HOLD        4.00x
Speculative decoding              8      $14,400        133ms         44ms    HOLD        4.00x
Routing + response cache          5       $9,000         60ms         18ms    HOLD        6.40x

32 GPUs → 5 GPUs · 6.4x cheaper · $57,600 → $9,000/month · SLO held · quality −0.31%
```

Runs on a laptop. No GPU, no network, no API key, ~15 seconds.

---

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt

.venv/bin/python demo.py              # the demo
.venv/bin/python bench/validate.py    # 11/11 published reference points
.venv/bin/python -m pytest tests/ -q  # 21 tests
```

In the demo, after the five stages: `l` load spike · `v` validation ·
`b` before-vLLM comparison · `r` real measured API numbers · `q` quit.

---

## What it actually models

The hard part of an inference cost model is that **latency must be an output,
not a parameter**. If you assume the latency, the model can only tell you what
you already believed. So this simulates the server at iteration granularity —
the same loop vLLM runs — and lets queueing and contention produce the latency.

### Real implementations (not fudge factors)

| Module | What it is |
|---|---|
| `infeng/kvcache.py` | PagedAttention block allocator — 16-token blocks, refcounted sharing, copy-on-write, recompute-based preemption. Plus a contiguous allocator in both its historical forms, for comparison. |
| `infeng/prefixcache.py` | Radix tree over resident KV blocks with **chained** block hashing (`hash(parent, tokens)`), LRU eviction of unpinned leaves only. |
| `infeng/engine.py` | Iteration-level scheduler: static batching vs continuous batching, chunked prefill, KV backpressure, admission control. |
| `infeng/router.py` | Cost-aware difficulty router with an explicit precision/recall trade and escalation charged honestly. |
| `infeng/responsecache.py` | Exact + near-duplicate (MinHash/LSH) response cache. |
| `infeng/speculative.py` | Draft/verify with the closed-form acceptance model. |

### Calibrated physics

`infeng/roofline.py` models the two regimes separately, because conflating them
is the root of most bad inference decisions:

- **Prefill** — one pass over many tokens. Big GEMMs, compute-bound. Sets TTFT.
- **Decode** — one pass per token per sequence. Streams the weights, so it is
  memory-bandwidth-bound. Sets TPOT. **Cost per step is nearly independent of
  batch size** — which is the entire economic argument for batching.

Parameter counts, KV bytes, and FLOPs are *derived from the model config*, not
hard-coded. GPU peaks are vendor-published. The only fitted numbers in the whole
model are achieved MFU and bandwidth efficiency, and they live in one dataclass
(`Efficiency`) so you can see and change them.

---

## Is any of it real?

```bash
$ .venv/bin/python bench/validate.py

Llama-3.1-70B parameter count                 70.60B     70.55B    -0.1%   PASS
70B FP16 KV cache per token                  320 KiB    320 KiB    +0.0%   PASS
70B TP4 weight-read floor per decode step    13.2 ms    13.2 ms    -0.3%   PASS
70B TP4 FP16 single-stream decode           55 tok/s   59 tok/s    +6.5%   PASS
70B TP4 FP16 prefill throughput         10,000 tok/s  9,535 tok/s  -4.7%   PASS
INT8 W8A8 decode speedup vs FP16               1.75x      1.81x    +3.7%   PASS
AWQ INT4 prefill time vs FP16 (expected >1)    1.12x      1.11x    -1.3%   PASS
PagedAttention KV waste (16-token blocks)    ≤ 4.00%      0.50%    +0.0%   PASS
...
11/11 passed · worst-case error 6.5%
```

Each check states its own tolerance, because reproducing architecture
arithmetic to 1% and real serving throughput to 30% are different kinds of
confidence and should not be reported as one number.

### Measured, not modeled

`infeng/realbench.py` makes **real Anthropic API calls** to check two of the
simulator's claims against a system with a billing department:

- **Prefix caching** — send a large shared prefix twice, read the real
  `cache_read_input_tokens` and the real cost delta off the response.
- **Routing** — same prompts, small vs large model, real latency and cost.

Record once, replay offline forever:

```bash
python -m infeng.realbench record   # ~8 calls, max_tokens=150, a few cents
python -m infeng.realbench show     # replays the committed cassette
```

> The system prompt is deliberately sized past **4096 tokens** — Claude Haiku
> 4.5's minimum cacheable prefix. Below it, caching silently does nothing and
> reports zero, which is the most common reason teams conclude caching "didn't
> work". The recorder asserts a cache write actually happened rather than
> trusting it.

---

## Reading the result honestly

**The baseline is stock vLLM, not a strawman.** An earlier version anchored on
hand-rolled static batching and produced a ~100x headline. Every factor in it
was individually defensible and the total was not credible — and it measured
against a configuration nobody in the room is running. Anchoring on
continuous-batching + paged KV gives **6.4x**, which is the number that is
actually actionable if you already run vLLM. Static batching survives as a
matched-fleet comparison (press `b`): same 32 GPUs, same traffic, only the
scheduler differs — 305 s vs 170 ms p95 TTFT.

**One lever deliberately does nothing.** Speculative decoding comes out at
8 → 8 GPUs with *worse* TPOT. That is not a bug: by that stage the batch is 98
sequences and the system is compute-bound, so the extra k+1 FLOPs per step
compete with work already queued. Speculation is free only while you are
memory-bound. A benchmark that quotes a speculative-decoding speedup without
quoting its batch size has told you nothing.

**What's modeled vs measured** is on slide 14 and in the docstrings. Achieved
MFU, draft acceptance rates, and published quantization quality deltas are
priors, not measurements of your cluster.

---

## Pointing it at your own traffic

The framework is the deliverable; the numbers are an example.

- **Workload** — `infeng/trace.py`. Archetype mix, prompt/output distributions,
  burstiness, shared-prefix structure, and Zipfian repeat rate are all knobs,
  because every cache number in this repo is a property of *your* traffic.
- **Model** — `infeng/model.py`. Add a `Model(...)`; params and KV bytes derive
  from the config.
- **Hardware and price** — `infeng/hardware.py`.
- **SLO** — `SLO(ttft_p95_ms=..., tpot_p95_ms=...)`.

---

## The order to actually do this in

0. **Instrument first.** p50/p95 TTFT and TPOT, tokens in/out, $/1M — per route.
1. **Measure your prompt:output ratio.** It decides whether you are prefill- or
   decode-bound, and therefore which half of the toolbox applies.
2. **Turn on what's free.** Continuous batching, paged KV, prefix caching, an
   exact response cache. Zero quality cost.
3. **Quantize — W8A8 before W4A16.** INT8/FP8 shrinks weights *and* KV *and*
   buys compute. W4A16 helps decode only and makes prefill slower.
4. **Route.** The biggest remaining lever is architectural, not kernel-level.
   Bound the under-route rate, escalate, and charge yourself for the retry.
5. **Only now, speculative decoding** — and only if you are latency-constrained
   at low batch.

---

## Repo layout

```
demo.py                    the stage demo
infeng/
  hardware.py  model.py  quant.py     profiles (published specs + config-derived math)
  roofline.py                         prefill/decode execution time
  kvcache.py   prefixcache.py         paged + contiguous allocators, radix cache
  responsecache.py  router.py  speculative.py
  engine.py                           iteration-level server simulation
  trace.py     metrics.py  config.py  workload, SLO/percentiles, the five stages
  realbench.py                        real API measurement + cassette
bench/
  validate.py                         11 published reference points
  deck_numbers.json                   generated: the deck reads this
tests/                                21 tests
scripts/
  snapshot_numbers.py                 simulator  → deck_numbers.json
  build_deck.py                       deck_numbers.json → PPTX
  lint_deck.py                        geometry/overflow lint (no renderer needed)
slides/                               the generated deck
```

The deck is generated from `bench/deck_numbers.json`, which is generated by the
simulator. Nothing on a slide is typed by hand, so the slides cannot disagree
with the demo running three feet away from them.

```bash
python scripts/snapshot_numbers.py && python scripts/build_deck.py && python scripts/lint_deck.py
```

---

MIT · Devang Sharma · [@idevangsharma](https://x.com/idevangsharma)
