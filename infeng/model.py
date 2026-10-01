"""Transformer model profiles, with parameter and KV-cache math derived from
architecture rather than hard-coded, so the numbers stay honest if you swap
in your own model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .quant import FP16, Quantization


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    hidden: int
    heads: int          # query heads
    kv_heads: int       # key/value heads (GQA: kv_heads < heads)
    head_dim: int
    intermediate: int   # FFN inner dim
    vocab: int
    tied_embeddings: bool = False

    # -- parameters -------------------------------------------------------
    @property
    def attn_params_per_layer(self) -> int:
        kv_dim = self.kv_heads * self.head_dim
        q_dim = self.heads * self.head_dim
        return (
            self.hidden * q_dim        # q_proj
            + self.hidden * kv_dim     # k_proj
            + self.hidden * kv_dim     # v_proj
            + q_dim * self.hidden      # o_proj
        )

    @property
    def ffn_params_per_layer(self) -> int:
        # SwiGLU: gate + up + down
        return 3 * self.hidden * self.intermediate

    @property
    def params_per_layer(self) -> int:
        return self.attn_params_per_layer + self.ffn_params_per_layer

    @property
    def embedding_params(self) -> int:
        n = self.vocab * self.hidden
        return n if self.tied_embeddings else 2 * n

    @property
    def total_params(self) -> int:
        return self.layers * self.params_per_layer + self.embedding_params

    # -- memory -----------------------------------------------------------
    def weight_bytes(self, quant: Quantization = FP16) -> float:
        return self.total_params * quant.weight_bytes_per_param

    def kv_bytes_per_token(self, quant: Quantization = FP16) -> float:
        """K and V, every layer, every KV head. This is the number that decides
        how many concurrent sequences fit on the box."""
        return (
            2                              # K and V
            * self.layers
            * self.kv_heads
            * self.head_dim
            * quant.kv_bytes_per_elem
        )

    # -- work -------------------------------------------------------------
    @property
    def flops_per_token(self) -> float:
        """Forward-pass FLOPs per token: 2 (MAC) x params, excluding embeddings
        (a lookup) but including the lm_head GEMM."""
        body = self.layers * self.params_per_layer
        head = self.vocab * self.hidden
        return 2.0 * (body + head)

    def attention_flops(self, seq_len: int, n_tokens: int) -> float:
        """Attention score+output FLOPs, which scale with context and are not
        captured by the parameter count. Matters for long prompts."""
        q_dim = self.heads * self.head_dim
        return 2.0 * 2.0 * self.layers * n_tokens * seq_len * q_dim


LLAMA_31_70B = Model(
    name="Llama-3.1-70B",
    layers=80, hidden=8192, heads=64, kv_heads=8, head_dim=128,
    intermediate=28672, vocab=128256,
)

LLAMA_31_8B = Model(
    name="Llama-3.1-8B",
    layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128,
    intermediate=14336, vocab=128256,
)

LLAMA_32_1B = Model(
    name="Llama-3.2-1B",
    layers=16, hidden=2048, heads=32, kv_heads=8, head_dim=64,
    intermediate=8192, vocab=128256, tied_embeddings=True,
)

CATALOG = {m.name: m for m in (LLAMA_31_70B, LLAMA_31_8B, LLAMA_32_1B)}
