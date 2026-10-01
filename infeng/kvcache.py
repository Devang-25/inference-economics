"""KV-cache allocators.

Two real implementations, so the memory-waste number in the demo is *measured
from the allocator* rather than asserted from a slide.

ContiguousAllocator  -- the pre-vLLM approach. Each sequence reserves a slab
                        sized for max_model_len at admission, because the
                        attention kernel needs contiguous K/V. Every token you
                        reserve but never generate is dead memory. This is
                        internal fragmentation, and it is enormous: reserving
                        4096 for a request that emits 180 tokens wastes 95% of
                        that slab.

PagedAllocator       -- PagedAttention. KV lives in fixed-size blocks (16
                        tokens) in a global pool, allocated on demand as the
                        sequence grows, with a block table doing the
                        virtual->physical indirection. Waste collapses to at
                        most one partial block per sequence. Blocks are
                        refcounted, so a shared prompt prefix is stored ONCE
                        across every sequence that shares it -- which is what
                        makes the prefix cache in prefixcache.py possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class OutOfKVMemory(Exception):
    """Raised when admission would exceed the KV pool. The scheduler treats this
    as backpressure, not as a crash -- see scheduler.py."""


@dataclass
class AllocatorStats:
    allocated_tokens: int = 0     # slots actually holding a real token
    reserved_tokens: int = 0      # slots taken out of the pool
    peak_reserved: int = 0
    preemptions: int = 0
    evictions: int = 0
    shared_tokens: int = 0        # tokens served from a refcount-shared block

    @property
    def waste_pct(self) -> float:
        """Deprecated: cumulative allocation vs peak reservation is not a
        meaningful ratio. Instantaneous waste is sampled per iteration by the
        engine instead -- see ReplicaResult.kv_waste_samples."""
        return 0.0


class ContiguousAllocator:
    """Contiguous KV, in the two forms it actually took in practice.

    reserve_mode="dynamic"   -- the static-batching era. The batch is one tensor,
                               rebuilt for every batch, so it only ever has to
                               be as long as the longest sequence in THAT batch.
                               Memory is not the problem here; scheduling is.

    reserve_mode="model_max" -- what happens the moment you go continuous while
                               keeping contiguous KV. Sequences now join and
                               leave asynchronously, so you cannot resize a
                               shared tensor underneath them: each slot must be
                               reserved for the full advertised context window
                               up front. Advertise 32K, reserve 32K, for a
                               900-token chat. This is the memory wall that
                               PagedAttention was built to remove, and it is why
                               continuous batching and paging shipped together.
    """

    kind = "contiguous"

    def __init__(self, capacity_tokens: int, max_model_len: int = 4096,
                 reserve_mode: str = "model_max"):
        self.capacity = capacity_tokens
        self.max_model_len = max_model_len
        self.reserve_mode = reserve_mode
        self.reserved = 0
        self._seqs: dict[int, int] = {}       # seq_id -> real tokens held
        self._slots: dict[int, int] = {}      # seq_id -> tokens reserved
        self.stats = AllocatorStats()

    @property
    def free_tokens(self) -> int:
        return self.capacity - self.reserved

    def _slot_size(self, prompt_len: int) -> int:
        if self.reserve_mode == "dynamic":
            return prompt_len
        return self.max_model_len

    def can_admit(self, prompt_len: int) -> bool:
        return self.free_tokens >= self._slot_size(prompt_len)

    def admit(self, seq_id: int, prompt_len: int) -> None:
        if not self.can_admit(prompt_len):
            raise OutOfKVMemory
        self._slots[seq_id] = self._slot_size(prompt_len)
        self.reserved += self._slots[seq_id]
        self._seqs[seq_id] = prompt_len
        self.stats.allocated_tokens += prompt_len
        self.stats.reserved_tokens = self.reserved
        self.stats.peak_reserved = max(self.stats.peak_reserved, self.reserved)

    def append(self, seq_id: int, n: int = 1) -> None:
        self._seqs[seq_id] += n
        self.stats.allocated_tokens += n
        if self.reserve_mode == "dynamic":
            # The batch tensor is grown to hold the longest member, so the
            # reservation tracks real length rather than the context window.
            grow = max(0, self._seqs[seq_id] - self._slots[seq_id])
            self._slots[seq_id] += grow
            self.reserved += grow

    def release(self, seq_id: int, keep_blocks: list[int] | None = None) -> None:
        del keep_blocks  # contiguous slabs are not block-addressable, nothing to keep
        if seq_id in self._seqs:
            self.reserved -= self._slots.pop(seq_id, self.max_model_len)
            del self._seqs[seq_id]
            self.stats.reserved_tokens = self.reserved

    def live_kv_tokens(self) -> int:
        """Tokens the attention kernel must actually read this step."""
        return sum(self._seqs.values())

    def waste_now_pct(self) -> float:
        """Reserved-but-unused KV, right now. For contiguous allocation this is
        the gap between max_model_len and what each sequence actually holds --
        reserving 32K for a 900-token chat wastes 97% of the slab."""
        if self.reserved <= 0:
            return 0.0
        return 100.0 * (1.0 - self.live_kv_tokens() / self.reserved)

    def cache_residency_pct(self) -> float:
        return 0.0

    def blocks_of(self, seq_id: int) -> list[int]:
        return []

    def has_room_to_grow(self, seq_id: int, n: int = 1) -> bool:
        return True

    def pin(self, block_ids: list[int]) -> None:
        return None

    def unpin(self, block_ids: list[int]) -> None:
        return None

    def preempt(self, seq_id: int) -> int:
        n = self._seqs.get(seq_id, 0)
        self.release(seq_id)
        self.stats.preemptions += 1
        return n


@dataclass
class _Block:
    block_id: int
    refcount: int = 1
    n_tokens: int = 0            # 0..block_size
    prefix_hash: int | None = None   # set when the block is a cacheable prefix block


class PagedAllocator:
    """PagedAttention: fixed-size blocks, on-demand growth, refcounted sharing."""

    kind = "paged"

    def __init__(self, capacity_tokens: int, block_size: int = 16,
                 max_model_len: int = 4096):
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.n_blocks = capacity_tokens // block_size
        self.capacity = self.n_blocks * block_size
        self._free: list[int] = list(range(self.n_blocks))
        self._blocks: dict[int, _Block] = {}
        self._tables: dict[int, list[int]] = {}   # seq_id -> block table
        self._lens: dict[int, int] = {}
        self.stats = AllocatorStats()

    # -- pool -------------------------------------------------------------
    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def free_tokens(self) -> int:
        return self.free_blocks * self.block_size

    @property
    def used_blocks(self) -> int:
        return self.n_blocks - self.free_blocks

    def _blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)   # ceil div

    def _alloc_block(self) -> _Block:
        if not self._free:
            raise OutOfKVMemory
        blk = _Block(self._free.pop())
        self._blocks[blk.block_id] = blk
        return blk

    def _free_block(self, block_id: int) -> None:
        blk = self._blocks[block_id]
        blk.refcount -= 1
        if blk.refcount <= 0:
            del self._blocks[block_id]
            self._free.append(block_id)

    # -- sequence lifecycle ----------------------------------------------
    def can_admit(self, prompt_len: int, shared_blocks: list[int] | None = None) -> bool:
        shared = len(shared_blocks or [])
        need = self._blocks_needed(prompt_len) - shared
        return self.free_blocks >= max(need, 0)

    def admit(self, seq_id: int, prompt_len: int,
              shared_blocks: list[int] | None = None) -> None:
        """Admit a sequence. `shared_blocks` are already-resident prefix blocks
        (from the prefix cache) that we bump a refcount on instead of
        re-materializing -- zero new memory, zero recompute."""
        shared_blocks = shared_blocks or []
        table: list[int] = []
        for bid in shared_blocks:
            self._blocks[bid].refcount += 1
            table.append(bid)
        shared_tokens = len(shared_blocks) * self.block_size
        remaining = max(prompt_len - shared_tokens, 0)

        need = self._blocks_needed(remaining)
        if self.free_blocks < need:
            for bid in table[len(shared_blocks):]:
                self._free_block(bid)
            for bid in shared_blocks:
                self._blocks[bid].refcount -= 1
            raise OutOfKVMemory

        left = remaining
        while left > 0:
            blk = self._alloc_block()
            blk.n_tokens = min(left, self.block_size)
            table.append(blk.block_id)
            left -= blk.n_tokens

        self._tables[seq_id] = table
        self._lens[seq_id] = prompt_len
        self.stats.allocated_tokens += remaining
        self.stats.shared_tokens += shared_tokens
        self.stats.reserved_tokens = self.used_blocks * self.block_size
        self.stats.peak_reserved = max(self.stats.peak_reserved,
                                       self.stats.reserved_tokens)

    def append(self, seq_id: int, n: int = 1) -> None:
        """Grow by n tokens, allocating a new block only when the last one fills.
        This is where paging pays: we never reserve ahead of the token."""
        table = self._tables[seq_id]
        for _ in range(n):
            last = self._blocks[table[-1]] if table else None
            if last is None or last.n_tokens >= self.block_size or last.refcount > 1:
                # copy-on-write: a shared tail block must be forked before we write
                blk = self._alloc_block()
                table.append(blk.block_id)
                last = blk
            last.n_tokens += 1
            self._lens[seq_id] += 1
            self.stats.allocated_tokens += 1
        self.stats.reserved_tokens = self.used_blocks * self.block_size
        self.stats.peak_reserved = max(self.stats.peak_reserved,
                                       self.stats.reserved_tokens)

    def has_room_to_grow(self, seq_id: int, n: int = 1) -> bool:
        table = self._tables.get(seq_id)
        if not table:
            return self.free_blocks > 0
        last = self._blocks[table[-1]]
        if last.n_tokens + n <= self.block_size and last.refcount == 1:
            return True
        return self.free_blocks >= self._blocks_needed(n)

    def release(self, seq_id: int, keep_blocks: list[int] | None = None) -> None:
        """Release a sequence. `keep_blocks` stay resident with an extra
        refcount because the prefix cache is holding them for reuse."""
        keep = set(keep_blocks or [])
        for bid in self._tables.pop(seq_id, []):
            if bid in keep:
                continue
            self._free_block(bid)
        self._lens.pop(seq_id, None)
        self.stats.reserved_tokens = self.used_blocks * self.block_size

    def preempt(self, seq_id: int) -> int:
        """Evict a running sequence's blocks back to the pool (recompute-based
        preemption, as vLLM does under pressure). Returns tokens reclaimed."""
        n = self._lens.get(seq_id, 0)
        self.release(seq_id)
        self.stats.preemptions += 1
        return n

    def blocks_of(self, seq_id: int) -> list[int]:
        return list(self._tables.get(seq_id, []))

    def pin(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid in self._blocks:
                self._blocks[bid].refcount += 1

    def unpin(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid in self._blocks:
                self._free_block(bid)
                self.stats.evictions += 1

    def live_kv_tokens(self) -> int:
        return sum(self._lens.values())

    def waste_now_pct(self) -> float:
        """Internal fragmentation of LIVE sequences, right now: the partial tail
        block each sequence carries.

        Deliberately measured over live sequences only. Blocks the prefix cache
        is holding for future reuse are resident on purpose -- counting them as
        "waste" would make a working cache look like a memory leak. They are
        reported separately as cache residency.
        """
        held = sum(len(t) for t in self._tables.values()) * self.block_size
        if held <= 0:
            return 0.0
        return 100.0 * max(0.0, 1.0 - self.live_kv_tokens() / held)

    def cache_residency_pct(self) -> float:
        """Share of the pool held by prefix-cache blocks with no live sequence
        attached -- memory deliberately spent on reuse."""
        live = set()
        for t in self._tables.values():
            live.update(t)
        parked = self.used_blocks - len(live)
        return 100.0 * max(0, parked) / self.n_blocks if self.n_blocks else 0.0

    @property
    def utilization_pct(self) -> float:
        return 100.0 * self.used_blocks / self.n_blocks if self.n_blocks else 0.0
