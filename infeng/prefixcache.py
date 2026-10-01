"""Radix prefix cache over KV blocks (automatic prefix caching).

The observation this exploits: in real traffic, prompts are not random. A RAG
service re-sends the same system prompt and the same retrieved chunks; an agent
re-sends a growing conversation where every turn shares the entire previous
turn as a prefix; a code assistant re-sends the same file header. That shared
prefix was already computed and is already sitting in HBM.

A block-level radix tree lets us find the longest already-resident prefix of an
incoming prompt and skip prefilling it entirely. Chained block hashing is what
makes this safe: a block's identity is
    hash(parent_block_hash, token_ids_in_this_block)
so a block only matches if the ENTIRE preceding context matches too. Hashing
the block's tokens alone would happily serve you someone else's KV.

The win lands on TTFT and on prefill compute, which is exactly the resource
that competes with decode for the GPU.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def block_hash(parent: int | None, token_ids: tuple[int, ...]) -> int:
    """Chained hash. Identity of a block includes its whole ancestry."""
    h = hashlib.blake2b(digest_size=8)
    h.update(b"\x00" if parent is None else parent.to_bytes(8, "little", signed=False))
    h.update(b"".join(t.to_bytes(4, "little") for t in token_ids))
    return int.from_bytes(h.digest(), "little")


@dataclass
class _Node:
    key: int
    block_id: int
    parent: "_Node | None" = None
    children: dict[int, "_Node"] = field(default_factory=dict)
    last_used: int = 0
    pinned: int = 0          # live sequences currently referencing this block


@dataclass
class PrefixCacheStats:
    lookups: int = 0
    hit_sequences: int = 0
    prompt_tokens_seen: int = 0
    prompt_tokens_hit: int = 0
    blocks_inserted: int = 0
    blocks_evicted: int = 0

    @property
    def token_hit_rate_pct(self) -> float:
        if not self.prompt_tokens_seen:
            return 0.0
        return 100.0 * self.prompt_tokens_hit / self.prompt_tokens_seen

    @property
    def sequence_hit_rate_pct(self) -> float:
        if not self.lookups:
            return 0.0
        return 100.0 * self.hit_sequences / self.lookups


class RadixPrefixCache:
    """LRU-evicted radix tree of resident KV blocks."""

    def __init__(self, block_size: int = 16, max_blocks: int = 0):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.root = _Node(key=0, block_id=-1)
        self._by_block: dict[int, _Node] = {}
        self._clock = 0
        self.stats = PrefixCacheStats()

    @property
    def size(self) -> int:
        return len(self._by_block)

    def _blockify(self, token_ids: list[int]) -> list[tuple[int, ...]]:
        bs = self.block_size
        n_full = len(token_ids) // bs
        return [tuple(token_ids[i * bs:(i + 1) * bs]) for i in range(n_full)]

    def lookup(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest resident prefix. Returns (block_ids, tokens_hit)."""
        self._clock += 1
        self.stats.lookups += 1
        self.stats.prompt_tokens_seen += len(token_ids)

        node = self.root
        parent_hash: int | None = None
        hit_blocks: list[int] = []
        for chunk in self._blockify(token_ids):
            key = block_hash(parent_hash, chunk)
            child = node.children.get(key)
            if child is None:
                break
            child.last_used = self._clock
            hit_blocks.append(child.block_id)
            node, parent_hash = child, key

        # Never serve the entire prompt from cache: the model needs at least one
        # token to attend *from*, so the last block is always recomputed.
        if hit_blocks and len(hit_blocks) * self.block_size >= len(token_ids):
            hit_blocks.pop()

        tokens_hit = len(hit_blocks) * self.block_size
        if tokens_hit:
            self.stats.hit_sequences += 1
            self.stats.prompt_tokens_hit += tokens_hit
        return hit_blocks, tokens_hit

    def insert(self, token_ids: list[int], block_ids: list[int]) -> list[int]:
        """Register a finished sequence's prompt blocks for reuse.
        Returns the block ids the cache now holds a reference on."""
        node = self.root
        parent_hash: int | None = None
        retained: list[int] = []
        for i, chunk in enumerate(self._blockify(token_ids)):
            if i >= len(block_ids):
                break
            key = block_hash(parent_hash, chunk)
            child = node.children.get(key)
            if child is None:
                child = _Node(key=key, block_id=block_ids[i], parent=node)
                node.children[key] = child
                self._by_block[child.block_id] = child
                self.stats.blocks_inserted += 1
                retained.append(child.block_id)
            child.last_used = self._clock
            node, parent_hash = child, key
        return retained

    def evict_to_fit(self, want_blocks: int) -> list[int]:
        """LRU-evict unpinned leaves until `want_blocks` can be freed.
        Only leaves are evictable -- dropping an interior block would orphan
        every descendant prefix that depends on it."""
        freed: list[int] = []
        if self.max_blocks <= 0:
            return freed
        while self.size + want_blocks > self.max_blocks:
            leaves = [n for n in self._by_block.values() if not n.children and not n.pinned]
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_used)
            victim.parent.children.pop(victim.key, None)
            del self._by_block[victim.block_id]
            freed.append(victim.block_id)
            self.stats.blocks_evicted += 1
        return freed

    def pin(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if b in self._by_block:
                self._by_block[b].pinned += 1

    def unpin(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if b in self._by_block:
                self._by_block[b].pinned = max(0, self._by_block[b].pinned - 1)
