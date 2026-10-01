#!/usr/bin/env python3
"""Inference Economics — live demo.

    python demo.py                 # the stage demo
    python demo.py --fast          # smaller trace, for dry runs
    python demo.py --validate      # just the validation suite
    python demo.py --real          # just the real measured API numbers

The question the whole demo answers is not "how fast is it" but:

    Given this traffic and this latency SLO, how many GPUs do I need?

Throughput numbers are easy to cherry-pick by quietly relaxing latency. Fleet
size at a fixed SLO is not, and it is denominated in the unit finance cares
about. Every stage is sized by searching replica counts until p95 TTFT and p95
TPOT both hold, and the dollar figure falls out of the fleet that survived.
"""
from __future__ import annotations

import argparse
import sys
import time

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, ".")

from infeng.config import STAGES, evaluate, legacy_comparison
from infeng.hardware import H100_SXM
from infeng.metrics import SLO
from infeng.model import LLAMA_31_70B
from infeng.quant import FP16
from infeng.roofline import Executor
from infeng.trace import generate

console = Console()

C_DIM = "grey58"
C_HEAD = "bold white"
C_GOOD = "bold green"
C_BAD = "bold red"
C_WARN = "yellow"
C_ACC = "bold cyan"


def rule(title: str) -> None:
    console.print()
    console.rule(f"[{C_ACC}]{title}", style="grey35")
    console.print()


def money(x: float) -> str:
    return f"${x:,.0f}"


# ---------------------------------------------------------------- act 1
def show_setup(w, slo: SLO) -> None:
    ex = Executor(H100_SXM, LLAMA_31_70B, FP16, tp=4)
    s = w.stats()

    t = Table.grid(padding=(0, 3))
    t.add_column(style=C_DIM, justify="right")
    t.add_column(style="white")
    t.add_row("workload", f"{s['requests']:,} requests over {w.duration_s:.0f}s "
                          f"= [bold]{w.arrival_rate_rps:.1f} req/s[/bold]")
    t.add_row("", f"{w.description}")
    t.add_row("prompt tokens", f"p50 {s['prompt_p50']:,}  p95 {s['prompt_p95']:,}  "
                               f"p99 {s['prompt_p99']:,}   [{C_DIM}](heavy-tailed, as real traffic is)[/]")
    t.add_row("output tokens", f"p50 {s['output_p50']:,}  p95 {s['output_p95']:,}")
    t.add_row("demand", f"{w.total_prompt_tokens / w.duration_s:,.0f} prefill tok/s  ·  "
                        f"{w.total_output_tokens / w.duration_s:,.0f} decode tok/s")
    t.add_row("", "")
    t.add_row("model", f"{LLAMA_31_70B.name}  ({LLAMA_31_70B.total_params/1e9:.0f}B params, "
                       f"GQA {LLAMA_31_70B.kv_heads}/{LLAMA_31_70B.heads} heads)")
    t.add_row("hardware", f"{H100_SXM.name} · TP=4 · ${H100_SXM.usd_per_hour:.2f}/GPU-hr")
    t.add_row("KV per token", f"{LLAMA_31_70B.kv_bytes_per_token()/1024:.0f} KiB  "
                              f"[{C_DIM}]→ {ex.max_kv_tokens:,} cached tokens fit per replica[/]")
    t.add_row("context window", "32,768 tokens (advertised — and therefore reserved)")
    t.add_row("", "")
    t.add_row("[bold]SLO[/bold]", f"[bold]p95 TTFT ≤ {slo.ttft_p95_ms:.0f} ms   ·   "
                                  f"p95 TPOT ≤ {slo.tpot_p95_ms:.0f} ms[/bold]")
    t.add_row("", f"[{C_DIM}]Fixed. Every optimization is scored as: how few GPUs hold this line?[/]")

    console.print(Panel(t, title="[bold]The workload, the box, and the contract[/bold]",
                        border_style="cyan", padding=(1, 2)))


# ---------------------------------------------------------------- waterfall
def waterfall_table(rows, ref) -> Table:
    """Deliberately few columns. A stage demo is read from the back of a room,
    so anything that wraps at 80 columns is worse than not showing it -- the
    per-stage detail panel carries the rest."""
    t = Table(box=None, pad_edge=False, padding=(0, 2))
    t.add_column("stage", style="white", no_wrap=True, width=26)
    t.add_column("GPUs", justify="right", width=5)
    t.add_column("$/month", justify="right", width=9)
    t.add_column("TTFT p95", justify="right", width=9)
    t.add_column("TPOT p95", justify="right", width=9)
    t.add_column("SLO", justify="center", width=5)
    t.add_column("vs stock", justify="right", width=8)

    for st, r in rows:
        cum = ref.usd_per_million_tokens / r.usd_per_million_tokens
        gpu_style = C_GOOD if r.gpus < ref.gpus else "white"
        t.add_row(
            ("[bold]" + st.title + "[/bold]") if st.reference else st.title,
            f"[{gpu_style}]{r.gpus}[/]",
            money(r.monthly_usd),
            f"{r.replica.ttft_p95:,.0f}ms",
            f"{r.replica.tpot_p95:.0f}ms",
            f"[{C_GOOD}]HOLD[/]" if r.slo_held else f"[{C_BAD}]MISS[/]",
            f"[{C_ACC}]{cum:.2f}x[/]" if cum > 1.001 else f"[{C_DIM}]1.00x[/]",
        )
    return t


def stage_detail(st, r, prev) -> Panel:
    body = Table.grid(padding=(0, 2))
    body.add_column(style=C_DIM, justify="right")
    body.add_column(style="white")
    body.add_row("lever", st.lever)

    rep = r.replica
    if prev is not None:
        d_gpu = prev.gpus - r.gpus
        if d_gpu > 0:
            body.add_row("fleet", f"[{C_GOOD}]{prev.gpus} → {r.gpus} GPUs  "
                                  f"(−{d_gpu}, {prev.gpus/r.gpus:.2f}x)[/]")
            body.add_row("saved", f"[{C_GOOD}]{money(prev.monthly_usd - r.monthly_usd)}/month[/]")
        else:
            body.add_row("fleet", f"[{C_WARN}]{prev.gpus} → {r.gpus} GPUs  (no reduction)[/]")

    body.add_row("live batch", f"{rep.avg_batch:.0f} sequences generating concurrently")
    if rep.kv_waste_pct:
        body.add_row("KV fragmentation", f"{rep.kv_waste_pct:.1f}% of live blocks")
    if rep.prefix_hit_pct:
        body.add_row("prefix cache", f"{rep.prefix_hit_pct:.1f}% of prompt tokens never recomputed")
    if rep.cache_residency_pct:
        body.add_row("cache residency", f"{rep.cache_residency_pct:.1f}% of KV pool held for reuse")
    if rep.spec_tokens_per_step:
        body.add_row("speculative", f"{rep.spec_accept_pct:.0f}% acceptance → "
                                    f"{rep.spec_tokens_per_step:.2f} tokens per verify step")
    if st.routing:
        d = r.detail
        body.add_row("routing", f"{d['small_share_pct']:.0f}% of traffic to the 8B  "
                                f"({d['gpus_big']} GPUs big + {d['gpus_small']} small)")
        body.add_row("under-routed", f"{d['under_route_pct']:.1f}%  "
                                     f"[{C_DIM}](escalated and re-answered on the 70B — and charged for)[/]")
        body.add_row("response cache", f"{rep.cache_hit_pct:.0f}% of requests never reached a GPU")
    if rep.preemptions:
        body.add_row("preemptions", f"{rep.preemptions}")

    body.add_row("", "")
    body.add_row("", f"[italic]{st.takeaway}[/italic]")
    colour = "green" if (prev is None or r.gpus < prev.gpus) else "yellow"
    return Panel(body, title=f"[bold]{st.title}[/bold]", border_style=colour, padding=(1, 2))


# ---------------------------------------------------------------- act 3
def show_legacy(w, slo, replicas) -> None:
    rule("Before we optimize: what your serving framework already bought you")
    with console.status("[cyan]running static batching and continuous+paged on the SAME fleet…"):
        lc = legacy_comparison(w, slo, replicas)
    a, b = lc["legacy"], lc["stock"]
    t = Table(box=None, padding=(0, 3))
    t.add_column("", style=C_DIM)
    t.add_column("static batching\n(pre-vLLM)", justify="right")
    t.add_column("continuous + paged\n(stock vLLM)", justify="right")
    t.add_row("p95 TTFT", f"[{C_BAD}]{a.ttft_p95/1000:,.0f} s[/]", f"[{C_GOOD}]{b.ttft_p95:,.0f} ms[/]")
    t.add_row("decode throughput", f"{a.output_tokens_per_s:,.0f} tok/s",
              f"[{C_GOOD}]{b.output_tokens_per_s:,.0f} tok/s[/]")
    t.add_row("live batch", f"{a.avg_batch:.0f}", f"{b.avg_batch:.0f}")
    console.print(t)
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"Identical hardware ({replicas * 4} GPUs), identical traffic, only the scheduler differs: "
            f"[bold]{b.output_tokens_per_s / max(a.output_tokens_per_s, 1e-9):.1f}x the throughput "
            f"and TTFT from {a.ttft_p95/1000:,.0f} seconds down to {b.ttft_p95:,.0f} milliseconds.[/bold]\n\n"
            f"[{C_DIM}]You get this for free by not writing your own serving loop. "
            f"It is not what this talk is about — everything after here is work you still have to do.[/]"),
        border_style="grey50", padding=(1, 2)))


def show_load_spike(w, slo, stage) -> None:
    rule("Act 3 — what happens when traffic doubles")
    from dataclasses import replace

    from infeng.config import _executors, _shard, size_fleet
    from infeng.engine import simulate_replica
    from infeng.trace import Workload

    # A spike is the SAME traffic arriving faster, so compress arrival times
    # rather than generating a new trace. Regenerating would change the request
    # mix as well as the rate and confound the two.
    factor = 2.0
    spike = Workload(
        name="spike",
        requests=[replace(r, arrival_s=r.arrival_s / factor) for r in w.requests],
        duration_s=w.duration_s / factor,
        description=f"{factor:.0f}x arrival rate, identical request mix",
    )
    n, _, _ = size_fleet(stage, w.requests, slo)
    console.print(f"[{C_DIM}]Same fleet, sized for normal load. Offered traffic goes "
                  f"[bold]{w.arrival_rate_rps:.0f} → {spike.arrival_rate_rps:.0f} req/s[/bold] "
                  f"with no warning. Autoscaling cannot help: loading a 70B onto cold GPUs "
                  f"takes minutes, and the spike is now.[/]\n")

    ex, draft = _executors(stage)
    configs = (
        ("serve everything", replace(stage.server, admission_control=False)),
        ("admission control", replace(stage.server, admission_control=True,
                                      max_num_seqs=96, max_queue_depth=8)),
    )
    rows = []
    for label, cfg in configs:
        with console.status(f"[cyan]{label} under {factor:.0f}x load…"):
            rows.append((label, simulate_replica(cfg, _shard(spike.requests, n), ex, slo, draft)))

    t = Table(box=None, padding=(0, 3))
    t.add_column("", style=C_DIM, width=20)
    t.add_column("p95 TTFT", justify="right")
    t.add_column("p95 TPOT", justify="right")
    t.add_column("served in SLO", justify="right")
    t.add_column("shed at door", justify="right")
    t.add_column("SLO", justify="center")
    for label, r in rows:
        held = slo.holds(r.ttft_p95, r.tpot_p95)
        served = r.requests_completed
        t.add_row(
            label,
            f"[{C_GOOD if r.ttft_p95 <= slo.ttft_p95_ms else C_BAD}]{r.ttft_p95:,.0f} ms[/]",
            f"[{C_GOOD if r.tpot_p95 <= slo.tpot_p95_ms else C_BAD}]{r.tpot_p95:.0f} ms[/]",
            f"{served:,}" if held else f"[{C_BAD}]0[/]",
            f"{r.requests_dropped:,}",
            f"[{C_GOOD}]HOLD[/]" if held else f"[{C_BAD}]MISS[/]",
        )
    console.print(t)
    console.print()
    console.print(Panel(Text.from_markup(
        "You cannot serve 2x the traffic on a 1x fleet. That is arithmetic, not engineering. "
        "The only decision available is [bold]what the failure looks like[/bold].\n\n"
        "Serve everything, and the queue never drains: every request waits tens of seconds, so "
        "[bold]nobody[/bold] gets a usable answer and you have converted a capacity problem into a "
        "total outage. Cap concurrency and shed at the door, and the requests you do admit are "
        "answered inside the SLO while the rest get an immediate, honest rejection their client "
        "can retry or degrade around.\n\n"
        f"[{C_DIM}]Note it is the concurrency cap, not the queue limit, that saves TPOT — a deep "
        f"queue with unbounded batch fixes time-to-first-token and then misses every token after "
        f"it. Degrading gracefully is a design decision; the default is degrading catastrophically.[/]"),
        border_style="yellow", padding=(1, 2)))


def show_validation() -> None:
    rule("Is any of this real? — validation against published benchmarks")
    from bench.validate import run_validation
    results = run_validation()
    t = Table(box=None, padding=(0, 2))
    t.add_column("published reference", style="white")
    t.add_column("source", style=C_DIM)
    t.add_column("reported", justify="right")
    t.add_column("this model", justify="right")
    t.add_column("error", justify="right")
    t.add_column("", justify="center")
    for r in results:
        ok = abs(r["error_pct"]) <= r["tolerance_pct"]
        t.add_row(r["name"], r["source"], r["reference"], r["modeled"],
                  f"{r['error_pct']:+.1f}%",
                  f"[{C_GOOD}]PASS[/]" if ok else f"[{C_BAD}]FAIL[/]")
    console.print(t)
    worst = max(abs(r["error_pct"]) for r in results)
    passed = sum(abs(r["error_pct"]) <= r["tolerance_pct"] for r in results)
    console.print()
    console.print(Panel(Text.from_markup(
        f"[bold]{passed}/{len(results)} reference points reproduced, worst-case error "
        f"{worst:.1f}%.[/bold]\n\n"
        f"[{C_DIM}]This is a calibrated capacity model, not a measurement of your cluster. "
        f"It is the same kind of model you would build to plan a fleet — and the point of "
        f"publishing the error bar is that you can check it rather than trust it.[/]"),
        border_style="cyan", padding=(1, 2)))


def show_real() -> None:
    rule("Measured, not modeled — real API economics")
    from infeng import realbench
    cas = realbench.load()
    if cas is None:
        console.print(f"[{C_WARN}]No cassette recorded yet.[/]  "
                      f"Run [bold]python -m infeng.realbench record[/bold] on good Wi-Fi first.")
        return
    s = realbench.summarize(cas)
    console.print(f"[{C_DIM}]Recorded {s['recorded_at']} against the real Anthropic API. "
                  f"Replayed offline — no network used right now.[/]\n")
    if "cache" in s:
        c = s["cache"]
        t = Table(box=None, padding=(0, 3))
        t.add_column("prefix caching", style=C_DIM)
        t.add_column(justify="right")
        t.add_row("shared prefix", f"{c['prefix_tokens']:,} tokens")
        t.add_row("served from cache on a hit", f"{c['tokens_served_from_cache']:,} tokens "
                                                f"({c['cache_hit_rate_pct']:.0f}%)")
        t.add_row("cost, uncached", f"${c['uncached_cost']:.6f}")
        t.add_row("cost, cache hit", f"[{C_GOOD}]${c['warm_cost']:.6f}[/]")
        t.add_row("cost reduction", f"[{C_GOOD}]{c['cost_reduction_pct']:.0f}%[/]")
        t.add_row("TTFT uncached → cached", f"{c['ttft_uncached_ms']:,.0f} ms → "
                                            f"[{C_GOOD}]{c['ttft_cached_ms']:,.0f} ms[/]")
        console.print(t)
        console.print()
    if "routing" in s:
        r = s["routing"]
        t = Table(box=None, padding=(0, 3))
        t.add_column("model routing", style=C_DIM)
        t.add_column(justify="right")
        t.add_row("same prompt on the small model", f"${r['small_cost']:.6f}  "
                                                    f"({r['small_ttft_ms']:,.0f} ms TTFT)")
        t.add_row("same prompt on the large model", f"${r['large_cost']:.6f}  "
                                                    f"({r['large_ttft_ms']:,.0f} ms TTFT)")
        t.add_row("cost ratio", f"[{C_ACC}]{r['cost_ratio']:.1f}x[/]")
        console.print(t)
    console.print()
    console.print(Panel(Text.from_markup(
        "Two of the simulator's claims, checked against a real API with a receipt: "
        "a shared prefix is nearly free to re-send, and tier is the biggest single "
        "cost lever. The mechanism is identical in your own server — "
        "[bold]this is your KV prefix cache and your router, with someone else's billing "
        "system doing the measuring.[/bold]"),
        border_style="cyan", padding=(1, 2)))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller trace for dry runs")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--no-pause", action="store_true", help="run straight through")
    args = ap.parse_args()

    if args.validate:
        show_validation(); return 0
    if args.real:
        show_real(); return 0

    console.clear()
    console.print(Align.center(Text.from_markup(
        "\n[bold cyan]INFERENCE ECONOMICS[/bold cyan]\n"
        "[white]Engineering LLMs for Cost, Latency, and Scale[/white]\n"
        f"[{C_DIM}]Devang Sharma · LLM Day SF 2026 · github.com/Devang-25/inference-economics[/]\n")))

    w = generate(n_requests=2000 if args.fast else 3000)
    slo = SLO(ttft_p95_ms=1000, tpot_p95_ms=50)

    rule("Act 1 — the setup")
    show_setup(w, slo)

    def pause(msg="[grey58]press Enter[/]"):
        if not args.no_pause:
            console.print(); console.input(msg + " ")

    pause()

    rows, ref, prev = [], None, None
    rule("Act 2 — five levers, each priced in GPUs")
    for st in STAGES:
        with console.status(f"[cyan]sizing the fleet for [bold]{st.title}[/bold] — "
                            f"searching replica counts until the SLO holds…"):
            t0 = time.perf_counter()
            r = evaluate(st, w, slo)
            dt = time.perf_counter() - t0
        if ref is None:
            ref = r
        rows.append((st, r))
        console.print(stage_detail(st, r, prev))
        console.print(waterfall_table(rows, ref))
        console.print(f"[{C_DIM}]  (sized in {dt:.1f}s)[/]")
        prev = r
        pause()

    final = rows[-1][1]
    rule("Act 4 — what the demo just proved")
    console.print(Panel(Align.center(Text.from_markup(
        f"[bold white]{ref.gpus} GPUs  →  {final.gpus} GPUs[/bold white]\n"
        f"[bold cyan]{ref.usd_per_million_tokens/final.usd_per_million_tokens:.1f}x cheaper[/bold cyan]\n\n"
        f"[white]{money(ref.monthly_usd)}/month  →  {money(final.monthly_usd)}/month[/white]\n"
        f"[bold green]{money(ref.monthly_usd - final.monthly_usd)}/month saved[/bold green]\n\n"
        f"[white]p95 TTFT {final.replica.ttft_p95:,.0f} ms  ·  "
        f"p95 TPOT {final.replica.tpot_p95:.1f} ms  ·  SLO held throughout[/white]\n"
        f"[{C_DIM}]quality delta {final.quality_delta_pct:+.2f}% — "
        f"the only stage that costs quality is the one you can tune[/]\n")),
        border_style="cyan", padding=(1, 4)))

    console.print()
    console.print(f"[{C_DIM}]  [bold]l[/bold] load spike   [bold]v[/bold] validation   "
                  f"[bold]b[/bold] before-vLLM comparison   [bold]r[/bold] real API numbers   "
                  f"[bold]q[/bold] quit[/]")
    if args.no_pause:
        return 0
    while True:
        try:
            k = console.input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
        if k == "q":
            return 0
        if k == "l":
            show_load_spike(w, slo, STAGES[-2])
        elif k == "v":
            show_validation()
        elif k == "b":
            show_legacy(w, slo, ref.replicas)
        elif k == "r":
            show_real()


if __name__ == "__main__":
    raise SystemExit(main())
