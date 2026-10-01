"""Response-level caching: the cheapest token is the one you never generate.

Two tiers, because they fail differently:

  Exact cache    -- normalized-text hash. Zero false positives, but brittle:
                    one changed character misses. Cheap enough to always run.

  Near-dup cache -- MinHash over character 5-grams with LSH banding, at a
                    deliberately conservative threshold. Catches the variants
                    that actually dominate production traffic: casing,
                    punctuation, whitespace, retries, and template-generated
                    prompts that differ by a token or two.

An honest boundary, because this is where cache demos usually oversell: this
tier is LEXICAL. Measured on sample pairs, near-duplicates score ~1.0,
true paraphrases ("how do I" vs "how can I") score ~0.5, and unrelated prompts
score 0.0. At the default 0.72 threshold we take the near-duplicates and
decline the paraphrases, because a semantic cache that answers the wrong
question is worse than no cache at all. Catching paraphrases needs an
embedding model -- swap one in behind `SemanticBackend` and raise recall, but
own the correctness risk that comes with it, scope it per tenant, and never
let it serve personalized or time-sensitive answers.

Dependency-free and deterministic: no embedding model, no network, same answer
on every run, which is what makes it demoable and CI-testable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    t = normalize(text)
    if len(t) <= k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _minhash(text: str, n_perm: int = 64) -> tuple[int, ...]:
    """MinHash signature. Deterministic: blake2b with a per-permutation salt."""
    sh = _shingles(text)
    if not sh:
        return tuple([0] * n_perm)
    sig = []
    for i in range(n_perm):
        salt = i.to_bytes(2, "little")
        sig.append(min(
            int.from_bytes(hashlib.blake2b(salt + s.encode(), digest_size=8).digest(), "little")
            for s in sh
        ))
    return tuple(sig)


def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


@dataclass
class ResponseCacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    semantic_candidates: int = 0
    tokens_saved_prompt: int = 0
    tokens_saved_output: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate_pct(self) -> float:
        return 100.0 * self.hits / self.lookups if self.lookups else 0.0


class ResponseCache:
    def __init__(self, threshold: float = 0.72, n_perm: int = 64,
                 bands: int = 16, max_entries: int = 50_000,
                 semantic: bool = True):
        self.threshold = threshold
        self.n_perm = n_perm
        self.bands = bands
        self.rows = n_perm // bands
        self.max_entries = max_entries
        self.semantic = semantic
        self._exact: dict[str, tuple[int, int]] = {}      # key -> (prompt_tok, out_tok)
        self._sigs: dict[str, tuple[int, ...]] = {}
        self._buckets: dict[tuple[int, bytes], list[str]] = {}
        self._order: list[str] = []
        self.stats = ResponseCacheStats()

    def _key(self, prompt: str) -> str:
        return hashlib.blake2b(normalize(prompt).encode(), digest_size=16).hexdigest()

    def _band_keys(self, sig: tuple[int, ...]):
        for b in range(self.bands):
            chunk = sig[b * self.rows:(b + 1) * self.rows]
            digest = hashlib.blake2b(
                b"".join(v.to_bytes(8, "little") for v in chunk), digest_size=8
            ).digest()
            yield (b, digest)

    def get(self, prompt: str) -> tuple[str, int, int] | None:
        """Returns (tier, saved_prompt_tokens, saved_output_tokens) or None."""
        self.stats.lookups += 1
        k = self._key(prompt)
        if k in self._exact:
            p, o = self._exact[k]
            self.stats.exact_hits += 1
            self.stats.tokens_saved_prompt += p
            self.stats.tokens_saved_output += o
            return ("exact", p, o)

        if not self.semantic:
            return None

        sig = _minhash(prompt, self.n_perm)
        seen: set[str] = set()
        for bk in self._band_keys(sig):
            for cand in self._buckets.get(bk, ()):
                if cand in seen:
                    continue
                seen.add(cand)
        self.stats.semantic_candidates += len(seen)
        best, score = None, 0.0
        for cand in seen:
            s = jaccard(sig, self._sigs[cand])
            if s > score:
                best, score = cand, s
        if best is not None and score >= self.threshold:
            p, o = self._exact[best]
            self.stats.semantic_hits += 1
            self.stats.tokens_saved_prompt += p
            self.stats.tokens_saved_output += o
            return ("semantic", p, o)
        return None

    def put(self, prompt: str, prompt_tokens: int, output_tokens: int) -> None:
        k = self._key(prompt)
        if k in self._exact:
            return
        self._exact[k] = (prompt_tokens, output_tokens)
        self._order.append(k)
        if self.semantic:
            sig = _minhash(prompt, self.n_perm)
            self._sigs[k] = sig
            for bk in self._band_keys(sig):
                self._buckets.setdefault(bk, []).append(k)
        while len(self._order) > self.max_entries:
            old = self._order.pop(0)
            self._exact.pop(old, None)
            sig = self._sigs.pop(old, None)
            if sig:
                for bk in self._band_keys(sig):
                    if bk in self._buckets and old in self._buckets[bk]:
                        self._buckets[bk].remove(old)
