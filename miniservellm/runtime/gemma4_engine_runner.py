"""Stage E engine runner: wires the Gemma4 eager forward into Stage5Engine.

Implements the :class:`ModelRunner` protocol (fresh/incremental prefill +
decode) on top of :class:`Gemma4EagerTextRunner` primitives and
:class:`Gemma4PagedKVCacheManager`. Each request in a step batch is processed
independently (batch=1 per request); cross-request batching kernels come
with the later performance stages.

Key semantics preserved from the official implementation:
- KV sharing: layers 15..34 never compute K/V; they gather the source
  layer's entry, which by construction already contains the current
  chunk/token because the source layer runs earlier in the layer loop.
- Sliding window: enforced by slicing the gathered K/V to the last
  ``sliding_window`` positions instead of building masks (equivalent for
  causal decoding and contiguous prefill chunks).
- PLE: recomputed per chunk from the chunk token ids (identity branch) and
  chunk embeddings (context branch), exactly like the official forward.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager
from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4TextWeights
from miniservellm.runtime.gemma4_runner import (
    SLIDING,
    Gemma4EagerTextRunner,
    _rms_norm,
)
from miniservellm.runtime.metadata import DecodeRequestMetadata, PrefillRequestMetadata
from miniservellm.runtime.model_interface import (
    DecodeModelOutput,
    PrefillModelOutput,
)
from miniservellm.scheduler.request import Request


class Gemma4EngineModelRunner(Gemma4EagerTextRunner):
    """Gemma4 runner behind the Stage5Engine ModelRunner protocol."""

    supports_cuda_graph = False

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: ModelConfig,
        weights: Gemma4TextWeights,
        kv_cache_manager: Gemma4PagedKVCacheManager,
    ):
        # Weights must already live on the target device/dtype (see
        # load_gemma4_text_weights); the eager runner forbids moving here.
        super().__init__(model_config, weights)
        self.engine_config = engine_config
        self.kv_cache_manager = kv_cache_manager
        # Dense per-request KV mirror (stage H): attention reads contiguous
        # tensors (same zero-copy pattern as the direct runner) while the
        # paged pages stay the durable/accounting storage. Mirror size is
        # ~18KB/token/request — negligible. Cleared on request finish.
        self._dense: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        engine_config  # pages remain authoritative for capacity accounting

    def _dense_for(self, request_id: str) -> dict:
        return self._dense.setdefault(request_id, {})

    def drop_dense(self, request_id: str) -> None:
        self._dense.pop(request_id, None)

    def _gc_dense(self) -> None:
        """Drop mirrors of requests no longer tracked by the cache manager."""
        live = self.kv_cache_manager.req_block_tables
        for rid in [rid for rid in self._dense if rid not in live]:
            del self._dense[rid]

    # ------------------------------------------------------------------
    # protocol surface
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.w.embed_tokens.device

    @property
    def dtype(self) -> torch.dtype:
        return self.w.embed_tokens.dtype

    def forward_fresh_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        return self._run_prefill(requests, metas)

    def forward_incremental_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        return self._run_prefill(requests, metas)

    def forward_decode(
        self, requests: List[Request], metas: List[DecodeRequestMetadata]
    ) -> DecodeModelOutput:
        logits_by_request: Dict[str, torch.Tensor] = {}
        for req, meta in zip(requests, metas):
            logits_by_request[req.request_id] = self._decode_one(req, meta)
        return DecodeModelOutput(logits_by_request=logits_by_request)

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------
    def _run_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        self._gc_dense()
        logits_by_request: Dict[str, torch.Tensor] = {}
        for req, meta in zip(requests, metas):
            if not meta.chunk_token_ids:
                continue
            logits = self._prefill_chunk(req, meta)
            if meta.chunk_end == req.total_prompt_tokens() and logits is not None:
                logits_by_request[req.request_id] = logits
        return PrefillModelOutput(logits_by_request=logits_by_request)

    def _prefill_chunk(
        self, req: Request, meta: PrefillRequestMetadata
    ) -> Optional[torch.Tensor]:
        w = self.w
        chunk_ids = meta.chunk_token_ids
        start = meta.context_len_before_chunk
        total = start + len(chunk_ids)
        input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.device)
        if start == 0:
            # Fresh prefill: reset any stale mirror for this request.
            self._dense.pop(req.request_id, None)
        dense = self._dense_for(req.request_id)

        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = self._ple(input_ids, x) if self.config.hidden_size_per_layer_input else None

        # Absolute positions for RoPE: [start, start+T)
        positions = torch.arange(start, total, device=self.device, dtype=torch.float32)

        for lw in w.layers:
            spec = lw.layer_spec
            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)

            num_q = spec.num_attention_heads
            head_dim = spec.head_dim
            q = (h @ lw.q_proj.t()).view(len(chunk_ids), num_q, head_dim)
            q = _rms_norm(q, lw.q_norm, self.eps)
            cos, sin = self._cos_sin_for(spec.attention_type, positions)
            q = self._apply_rope(q, cos, sin)

            source = self.kv_source.get(spec.layer_idx)
            if source is not None:
                # Shared layer: K/V come from the source layer's mirror. The
                # source layer (< this index) already stored the current chunk.
                k, v = dense[source]
            else:
                k = (h @ lw.k_proj.t()).view(len(chunk_ids), spec.num_key_value_heads, head_dim)
                k = _rms_norm(k, lw.k_norm, self.eps)
                k = self._apply_rope(k, cos, sin)
                v = (h @ lw.v_proj.t()).view(len(chunk_ids), spec.num_key_value_heads, head_dim)
                v = _rms_norm(v, lw.v_norm, self.eps)
                self.kv_cache_manager.write_kv_for_tokens(
                    spec.layer_idx, meta.write_slots, k, v
                )
                prev = dense.get(spec.layer_idx)
                if prev is None:
                    dense[spec.layer_idx] = (k, v)
                else:
                    dense[spec.layer_idx] = (
                        torch.cat([prev[0], k], dim=0),
                        torch.cat([prev[1], v], dim=0),
                    )
                k, v = dense[spec.layer_idx]
            # NOTE: no window slicing here — prefill attention uses the full
            # gathered K/V with the sliding mask built over absolute
            # positions (_prefill_mask). Slicing would misalign K with the
            # mask when the chunk spans multiple windows.

            scores = torch.matmul(
                q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)
            ).float()
            scores = scores + self._prefill_mask(spec.attention_type, start, total, len(chunk_ids))
            probs = torch.softmax(scores, dim=-1).to(v.dtype)
            attn = torch.matmul(probs, v.transpose(0, 1))
            h = attn.transpose(0, 1).reshape(len(chunk_ids), num_q * head_dim) @ lw.o_proj.t()

            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            x = residual + h

            residual = x
            h = _rms_norm(x, lw.pre_feedforward_layernorm, self.eps)
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            x = residual + h

            if per_layer_inputs is not None:
                residual = x
                h = torch.nn.functional.gelu(x @ lw.per_layer_input_gate.t(), approximate="tanh")
                h = h * per_layer_inputs[:, spec.layer_idx, :]
                h = h @ lw.per_layer_projection.t()
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = residual + h

            x = x * lw.layer_scalar

        x = _rms_norm(x, w.final_norm, self.eps)
        last_hidden = x[-1:]
        logits = last_hidden @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _decode_one(self, req: Request, meta: DecodeRequestMetadata) -> torch.Tensor:
        w = self.w
        dense = self._dense_for(req.request_id)
        input_ids = torch.tensor([meta.input_token_id], dtype=torch.long, device=self.device)
        position = meta.query_position
        ctx_len = meta.context_len

        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = None
        if self.config.hidden_size_per_layer_input:
            ple_dim = self.config.hidden_size_per_layer_input
            identity = w.embed_tokens_per_layer[input_ids] * float(ple_dim) ** 0.5
            identity = identity.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = (x @ w.per_layer_model_projection.t()) * float(self.config.hidden_size) ** -0.5
            ctx = ctx.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
            per_layer_inputs = (ctx + identity) * (2.0 ** -0.5)

        positions = torch.tensor([float(position)], device=self.device, dtype=torch.float32)
        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            q = (h @ lw.q_proj.t()).view(1, num_q, head_dim)
            q = _rms_norm(q, lw.q_norm, self.eps)
            cos, sin = self._cos_sin_for(spec.attention_type, positions)
            q = self._apply_rope(q, cos, sin)

            source = self.kv_source.get(spec.layer_idx)
            if source is not None:
                k, v = dense[source]
            else:
                k = (h @ lw.k_proj.t()).view(1, spec.num_key_value_heads, head_dim)
                k = _rms_norm(k, lw.k_norm, self.eps)
                k = self._apply_rope(k, cos, sin)
                v = (h @ lw.v_proj.t()).view(1, spec.num_key_value_heads, head_dim)
                v = _rms_norm(v, lw.v_norm, self.eps)
                self.kv_cache_manager.write_kv_for_tokens(
                    spec.layer_idx, [meta.write_slot], k, v
                )
                prev = dense.get(spec.layer_idx)
                if prev is None:
                    dense[spec.layer_idx] = (k, v)
                else:
                    dense[spec.layer_idx] = (
                        torch.cat([prev[0], k], dim=0),
                        torch.cat([prev[1], v], dim=0),
                    )
                k, v = dense[spec.layer_idx]
            k, v = self._window_slice(spec, k, v)

            scores = torch.matmul(
                q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)
            )
            probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            attn = torch.matmul(probs, v.transpose(0, 1))
            h = attn.transpose(0, 1).reshape(1, num_q * head_dim) @ lw.o_proj.t()

            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            x = residual + h

            residual = x
            h = _rms_norm(x, lw.pre_feedforward_layernorm, self.eps)
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            x = residual + h

            if per_layer_inputs is not None:
                residual = x
                h = torch.nn.functional.gelu(x @ lw.per_layer_input_gate.t(), approximate="tanh")
                h = h * per_layer_inputs[:, spec.layer_idx, :]
                h = h @ lw.per_layer_projection.t()
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = residual + h

            x = x * lw.layer_scalar

        x = _rms_norm(x, w.final_norm, self.eps)
        logits = x @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _cos_sin_for(self, attention_type: str, positions: torch.Tensor):
        """cos/sin for arbitrary absolute positions [T, D]."""
        inv_freq = self.inv_freq[attention_type]
        freqs = positions[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def _window_slice(self, spec, k: torch.Tensor, v: torch.Tensor):
        if spec.attention_type == SLIDING and self.window and k.shape[0] > self.window:
            return k[-self.window:], v[-self.window:]
        return k, v

    def _prefill_mask(
        self, attention_type: str, start: int, total: int, chunk_len: int
    ) -> torch.Tensor:
        """Additive mask [chunk_len, total] over absolute positions.

        Query i sits at absolute position start+i and may attend key j iff
        j <= start+i (causal), restricted to the sliding window when the
        layer is a sliding layer.
        """
        device = self.device
        q_abs = torch.arange(start, total, device=device)
        k_abs = torch.arange(0, total, device=device)
        dist = q_abs[:, None] - k_abs[None, :]
        if attention_type == SLIDING:
            allowed = (dist >= 0) & (dist < self.window)
        else:
            allowed = dist >= 0
        mask = torch.zeros(chunk_len, total, device=device, dtype=torch.float32)
        return mask.masked_fill(~allowed, torch.finfo(torch.float32).min)
