"""CUDA Graph-accelerated batch=1 greedy decode (C1 optimization).

Captures the full decode step (embed -> 24 layers -> lm_head) as a CUDA
graph and replays it per step, eliminating kernel launch overhead.

Expected improvement: 10-30% for batch=1 decode.
"""

from __future__ import annotations

from typing import Optional

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.nn_ops import (
    apply_rope,
    decode_paged_attention,
    fused_add_rms_norm,
    linear,
    rms_norm,
    silu_and_mul,
)


class CudaGraphBatch1Runner:
    """CUDA Graph batch=1 greedy decode wrapper.

    Usage::

        runner = CudaGraphBatch1Runner(model_runner, max_context=512)
        runner.capture(first_token, ctx_len, block_table)
        for _ in range(max_new_tokens):
            logits = runner.step(next_token, ctx_len, block_table)
            next_token = logits.argmax(dim=-1).item()
            ctx_len += 1  # update block_table as blocks are consumed
    """

    def __init__(
        self,
        model_runner: TransformerModelRunner,
        max_context: int = 512,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CudaGraphBatch1Runner requires CUDA")

        self.runner = model_runner
        self._cfg: EngineConfig = model_runner.engine_config
        self._mc: ModelConfig = model_runner.model_config
        self._max_ctx = max_context
        self._device = self._cfg.device
        self._vocab_size = self._mc.vocab_size
        self._hidden_size = self._mc.hidden_size
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

        # Static tensors allocated once, reused across steps
        self._static_tokens = torch.zeros(
            1, 1, dtype=torch.long, device=self._device
        )
        self._static_block_table = torch.zeros(
            1, max_context, dtype=torch.int32, device=self._device
        )
        self._static_context_lens = torch.zeros(
            1, dtype=torch.int32, device=self._device
        )
        self._static_logits = torch.empty(
            1, self._vocab_size, dtype=self._cfg.dtype, device=self._device
        )

    # ---- graphable attention sub-layer ----

    def _graphable_attention(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        layer_weights,
        kv_cache_manager,
    ) -> torch.Tensor:
        """Tensor-only attention sub-layer: norm + qkv + rope + kv write + paged attn + o_proj."""
        x_normed = rms_norm(
            hidden, layer_weights.input_layernorm, self._mc.rms_norm_eps
        )

        qkv = linear(x_normed, layer_weights.qkv_proj, layer_weights.qkv_proj_bias)
        q_dim = self._mc.num_attention_heads * self._mc.head_dim
        kv_dim = self._mc.num_key_value_heads * self._mc.head_dim
        q_raw, k_new, v_new = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q_raw.view(1, self._mc.num_attention_heads, self._mc.head_dim)
        k_new = k_new.view(1, self._mc.num_key_value_heads, self._mc.head_dim)
        v_new = v_new.view(1, self._mc.num_key_value_heads, self._mc.head_dim)

        # RoPE — reuse nn_ops.apply_rope (graph-capturable, pure tensor ops)
        rope_pos = self._static_context_lens.to(torch.long)
        q, k_new = apply_rope(q, k_new, rope_pos, self.runner.rope_cos, self.runner.rope_sin)

        # KV write to paged cache (in-place, graph-capturable)
        k_cache = kv_cache_manager.k_cache[layer_idx]
        v_cache = kv_cache_manager.v_cache[layer_idx]
        bs = k_cache.size(1)  # block_size
        pos_long = rope_pos  # already long
        wb = (pos_long // bs).view(-1)
        wo = (pos_long % bs).view(-1)
        kv_head_range = torch.arange(kv_dim, device=self._device, dtype=torch.long)

        k_cache.index_put_(
            (wb, wo.expand(kv_dim), kv_head_range),
            k_new.view(-1),
        )
        v_cache.index_put_(
            (wb, wo.expand(kv_dim), kv_head_range),
            v_new.view(-1),
        )

        # Paged attention
        attn_out = decode_paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=self._static_block_table,
            context_lens=self._static_context_lens.to(torch.int32),
        )

        attn_out = attn_out.reshape(1, self._hidden_size)
        return linear(attn_out, layer_weights.o_proj, layer_weights.o_proj_bias)

    # ---- full forward ----

    def _graphable_forward(self) -> torch.Tensor:
        """Full decode forward (embed -> layers -> norm -> lm_head), pure tensor ops."""
        hidden = self.runner.weights.embed_tokens[self._static_tokens]
        kv_mgr = self.runner.kv_cache_manager

        for layer_idx in range(self._mc.num_hidden_layers):
            lw = self.runner.weights.layers[layer_idx]
            residual = hidden

            attn_out = self._graphable_attention(hidden, layer_idx, lw, kv_mgr)

            x_normed, hidden = fused_add_rms_norm(
                attn_out, residual,
                lw.post_attention_layernorm, self._mc.rms_norm_eps,
            )

            gate_up = linear(x_normed, lw.gate_up_proj)
            gate, up = gate_up.split(self._mc.intermediate_size, dim=-1)
            act = silu_and_mul(gate, up)
            hidden = hidden + linear(act, lw.down_proj)

        hidden = rms_norm(
            hidden, self.runner.weights.final_norm, self._mc.rms_norm_eps
        )
        return linear(hidden[:, -1:, :], self.runner.weights.lm_head)

    # ---- public API ----

    @property
    def is_captured(self) -> bool:
        return self._captured

    def capture(
        self,
        first_token: int,
        context_len: int,
        block_table: torch.Tensor,
    ) -> None:
        """Warmup then capture the decode forward as a CUDA graph.

        Args:
            first_token: first generated token id.
            context_len: KV context length after prefill.
            block_table: [1, max_blocks] int32 block table.
        """
        if self._captured:
            return

        self._static_tokens.copy_(
            torch.tensor([[first_token]], device=self._device, dtype=torch.long)
        )
        self._static_context_lens.copy_(
            torch.tensor([context_len], device=self._device, dtype=torch.int32)
        )
        bt = block_table.to(torch.int32)
        n = bt.shape[1]
        self._static_block_table[:, :n].copy_(bt)

        # Warmup (3 iterations to stabilize GPU clock)
        for _ in range(3):
            self._graphable_forward()
        torch.cuda.synchronize()

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_logits = self._graphable_forward()
        torch.cuda.synchronize()

        self._captured = True

    def step(
        self,
        token_id: int,
        context_len: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Replay captured graph with new token and KV state.

        Returns:
            logits: [1, vocab_size] tensor (fp16/bf16, on CUDA).
        """
        if not self._captured:
            raise RuntimeError("call capture() before step()")
        assert self._graph is not None

        self._static_tokens.copy_(
            torch.tensor([[token_id]], device=self._device, dtype=torch.long)
        )
        self._static_context_lens.copy_(
            torch.tensor([context_len], device=self._device, dtype=torch.int32)
        )
        bt = block_table.to(torch.int32)
        n = bt.shape[1]
        self._static_block_table[:, :n].copy_(bt)

        self._graph.replay()
        return self._static_logits
