"""Gemma4 E2B text runner: batch=1, no-cache, PyTorch eager.

Implements the official Gemma4 text forward as a reference for the future
optimized runner:

- scaled word embedding (``sqrt(hidden_size)``);
- Per-Layer Embeddings (PLE): token-identity branch + context-aware
  projection branch combined with ``1/sqrt(2)``;
- four main RMSNorms per layer plus per-head q/k norm and a scale-free
  v-norm;
- sliding attention (head_dim=256, theta=10k) alternating with full
  attention (head_dim=512, proportional RoPE, 25% partial rotary);
- KV sharing: layers 15..34 reuse post-processed K/V from the last
  non-shared layer of the same attention type;
- GELU-tanh gated MLP (double-wide in the KV-shared tail);
- final logit soft-capping.

Generation is greedy and recomputes the full sequence each step; a
layer-aware cache and incremental decode come with the later cache stage.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import torch

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4TextWeights

SLIDING = "sliding_attention"
FULL = "full_attention"


def _rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """Official Gemma4RMSNorm: fp32 compute, optional scale, cast back."""
    x32 = x.float()
    normed = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _build_inv_freq(spec, head_dim: int) -> torch.Tensor:
    """Sliding RoPE uses the default formula; full layers use proportional.

    Proportional RoPE keeps the encoding at full head_dim width: the rotated
    part covers ``partial_rotary_factor * head_dim // 2`` frequencies and the
    remaining slots use inv_freq=0, which yields cos=1/sin=0 (identity).
    """
    theta = spec.rope_theta
    if spec.rope_type == "proportional":
        # partial_rotary_factor is folded into spec.rotary_dim by the config
        # converter (rotary_dim = head_dim * factor); the rotated half-width
        # is rotary_dim // 2 over the FULL head_dim exponent denominator.
        rotated = spec.rotary_dim // 2
        nope = head_dim // 2 - rotated
        inv_rot = 1.0 / (
            theta ** (torch.arange(0, 2 * rotated, 2, dtype=torch.int64).float() / head_dim)
        )
        inv = torch.cat([inv_rot, torch.zeros(nope)]) if nope > 0 else inv_rot
    else:
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    return inv


class Gemma4EagerTextRunner:
    """Reference eager forward for Gemma4 text-only inference."""

    def __init__(
        self,
        model_config: ModelConfig,
        weights: Gemma4TextWeights,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.config = model_config
        self.w = weights
        self.eps = model_config.rms_norm_eps
        self.softcap = model_config.final_logit_softcapping
        self.window = model_config.sliding_window

        if device is not None or dtype is not None:
            raise ValueError("Move weights before constructing the runner; device/dtype are fixed")

        self.inv_freq: dict[str, torch.Tensor] = {}
        for spec in (lw.layer_spec for lw in weights.layers):
            if spec.attention_type in self.inv_freq:
                continue
            inv = _build_inv_freq(spec, spec.head_dim).to(weights.embed_tokens.device)
            self.inv_freq[spec.attention_type] = inv

        # KV sharing: shared layers reuse K/V from the last non-shared layer
        # of the same attention type (official rule), i.e. layer 13 for
        # sliding and layer 14 for full with E2B's num_kv_shared_layers=20.
        first_shared = model_config.num_hidden_layers - model_config.num_kv_shared_layers
        prev_types = [lw.layer_spec.attention_type for lw in weights.layers[:first_shared]]
        self.kv_source: dict[int, int] = {}
        for lw in weights.layers[first_shared:]:
            spec = lw.layer_spec
            source = len(prev_types) - 1 - prev_types[::-1].index(spec.attention_type)
            self.kv_source[spec.layer_idx] = source
        self.storing_layers = set(self.kv_source.values())

        # Layer-aware KV cache (stage D). Stores post-norm post-rope K/V for
        # every non-shared layer; shared layers read their source layer's
        # entry. Shapes: k/v = [seq_len, num_kv_heads, head_dim].
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._seq_len = 0

    def reset_cache(self) -> None:
        self._cache.clear()
        self._seq_len = 0

    # ------------------------------------------------------------------
    # primitives
    # ------------------------------------------------------------------
    def _cos_sin(self, attention_type: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[attention_type]
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [T, H, D]; cos/sin: [T, D]. Official implementation casts
        # cos/sin to x.dtype before applying (modeling_gemma4.py rotary
        # forward), keeping q/k in the model dtype.
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]

    def _mask(self, attention_type: str, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        device = self.w.embed_tokens.device
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]
        if attention_type == SLIDING:
            allowed = (dist >= 0) & (dist < self.window)
        else:
            allowed = dist >= 0
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _attention(self, h: torch.Tensor, layer_idx: int, shared_kv: dict, capture: bool = False) -> torch.Tensor:
        spec = self.w.layers[layer_idx].layer_spec
        lw = self.w.layers[layer_idx]
        num_q = spec.num_attention_heads
        num_kv = spec.num_key_value_heads
        head_dim = spec.head_dim

        q = (h @ lw.q_proj.t()).view(h.shape[0], num_q, head_dim)
        q = _rms_norm(q, lw.q_norm, self.eps)
        cos, sin = self._cos_sin(spec.attention_type, h.shape[0])
        q = self._apply_rope(q, cos, sin)

        if layer_idx in self.kv_source:
            k, v = shared_kv[self.kv_source[layer_idx]]
        else:
            k = (h @ lw.k_proj.t()).view(h.shape[0], num_kv, head_dim)
            k = _rms_norm(k, lw.k_norm, self.eps)
            k = self._apply_rope(k, cos, sin)
            v = (h @ lw.v_proj.t()).view(h.shape[0], num_kv, head_dim)
            v = _rms_norm(v, lw.v_norm, self.eps)  # scale-free: v_norm has no weight
            if layer_idx in self.storing_layers:
                # Last non-shared layer of this type: shared layers 15..34
                # reuse these post-processed states within this forward.
                shared_kv[layer_idx] = (k, v)
            if capture:
                # Every non-shared layer stores its own cache entry.
                self._cache[layer_idx] = (k, v)

        # q: [T, H, D] -> [H, T, D]; k/v: [Tk, KV, D] -> [KV, Tk, D]
        scores = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2))
        # Official scaling is 1.0 (no 1/sqrt(head_dim)).
        scores = scores.float() + self._mask(spec.attention_type, h.shape[0], torch.float32)[None]
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        attn = torch.matmul(probs, v.transpose(0, 1))  # [H, T, D]
        return attn.transpose(0, 1).reshape(h.shape[0], num_q * head_dim) @ lw.o_proj.t()

    def _mlp(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        lw = self.w.layers[layer_idx]
        gate = torch.nn.functional.gelu(h @ lw.gate_proj.t(), approximate="tanh")
        up = h @ lw.up_proj.t()
        return (gate * up) @ lw.down_proj.t()

    def _ple(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """PLE = (context projection + token identity) * 1/sqrt(2)."""
        w = self.w
        ple_dim = self.config.hidden_size_per_layer_input
        num_layers = self.config.num_hidden_layers
        identity = w.embed_tokens_per_layer[input_ids] * float(ple_dim) ** 0.5
        identity = identity.view(input_ids.shape[0], num_layers, ple_dim)
        ctx = (inputs_embeds @ w.per_layer_model_projection.t()) * float(self.config.hidden_size) ** -0.5
        ctx = ctx.view(input_ids.shape[0], num_layers, ple_dim)
        ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
        return (ctx + identity) * (2.0 ** -0.5)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence eager forward. Returns fp32 logits [T, vocab]."""
        w = self.w
        input_ids = input_ids.to(w.embed_tokens.device)
        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5

        per_layer_inputs = self._ple(input_ids, x) if self.config.hidden_size_per_layer_input else None
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for lw in w.layers:
            spec = lw.layer_spec
            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            h = self._attention(h, spec.layer_idx, shared_kv)
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
        return logits.float()

    def _cos_sin_at(self, attention_type: str, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for a single position (incremental decode)."""
        inv_freq = self.inv_freq[attention_type]
        freqs = float(position) * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def _cached_kv_for_attention(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V for decode attention, window-sliced for sliding layers."""
        spec = self.w.layers[layer_idx].layer_spec
        source = self.kv_source.get(layer_idx, layer_idx)
        k, v = self._cache[source]
        if spec.attention_type == SLIDING and k.shape[0] > self.window:
            return k[-self.window:], v[-self.window:]
        return k, v

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Cache-aware prompt forward. Returns fp32 logits for the last token."""
        w = self.w
        self.reset_cache()
        input_ids = input_ids.to(w.embed_tokens.device)
        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5

        per_layer_inputs = self._ple(input_ids, x) if self.config.hidden_size_per_layer_input else None
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for lw in w.layers:
            spec = lw.layer_spec
            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            h = self._attention(h, spec.layer_idx, shared_kv, capture=True)
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

        self._seq_len = int(input_ids.shape[0])
        x = _rms_norm(x, w.final_norm, self.eps)
        logits = x[-1:] @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    @torch.inference_mode()
    def decode_step(self, token_id: int) -> torch.Tensor:
        """One incremental decode step. Returns fp32 logits [vocab]."""
        w = self.w
        ids = torch.tensor([token_id], dtype=torch.long, device=w.embed_tokens.device)
        x = w.embed_tokens[ids] * float(self.config.hidden_size) ** 0.5  # [1, H]

        per_layer_inputs = None
        if self.config.hidden_size_per_layer_input:
            ple_dim = self.config.hidden_size_per_layer_input
            num_layers = self.config.num_hidden_layers
            identity = w.embed_tokens_per_layer[ids] * float(ple_dim) ** 0.5
            identity = identity.view(1, num_layers, ple_dim)
            ctx = (x @ w.per_layer_model_projection.t()) * float(self.config.hidden_size) ** -0.5
            ctx = ctx.view(1, num_layers, ple_dim)
            ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
            per_layer_inputs = (ctx + identity) * (2.0 ** -0.5)

        position = self._seq_len
        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            q = (h @ lw.q_proj.t()).view(1, num_q, head_dim)
            q = _rms_norm(q, lw.q_norm, self.eps)
            cos, sin = self._cos_sin_at(spec.attention_type, position)
            q = self._apply_rope(q, cos, sin)  # [1, H, D]

            if spec.layer_idx in self.kv_source:
                pass  # K/V come from the source layer's cache entry below
            else:
                k = (h @ lw.k_proj.t()).view(1, spec.num_key_value_heads, head_dim)
                k = _rms_norm(k, lw.k_norm, self.eps)
                k = self._apply_rope(k, cos, sin)
                v = (h @ lw.v_proj.t()).view(1, spec.num_key_value_heads, head_dim)
                v = _rms_norm(v, lw.v_norm, self.eps)
                # Every non-shared layer keeps its own cache entry; shared
                # layers (15..34) read their source layer's entry instead.
                prev = self._cache.get(spec.layer_idx)
                if prev is None:
                    self._cache[spec.layer_idx] = (k, v)
                else:
                    self._cache[spec.layer_idx] = (
                        torch.cat([prev[0], k], dim=0),
                        torch.cat([prev[1], v], dim=0),
                    )

            k, v = self._cached_kv_for_attention(spec.layer_idx)
            # q: [1, H, D] -> [H, 1, D]; k: [Tk, KV, D] -> [KV, D, Tk]
            scores = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2))
            probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            attn = torch.matmul(probs, v.transpose(0, 1))  # [H, 1, D]
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

        self._seq_len += 1
        x = _rms_norm(x, w.final_norm, self.eps)
        logits = x @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    @torch.inference_mode()
    def generate_cached(
        self,
        prompt_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
    ) -> tuple[list[int], dict]:
        """Greedy generation with incremental decode; returns (new_ids, stats)."""
        prompt_ids = [int(t) for t in prompt_ids]
        generated: list[int] = []
        started = time.perf_counter()
        logits = self.prefill(torch.tensor(prompt_ids, dtype=torch.long))
        prefill_seconds = time.perf_counter() - started
        decode_seconds = 0.0
        next_id = int(logits.argmax())
        ids = list(prompt_ids)
        ids.append(next_id)
        generated.append(next_id)
        while len(generated) < max_new_tokens and next_id not in eos_token_ids:
            step_started = time.perf_counter()
            logits = self.decode_step(next_id)
            decode_seconds += time.perf_counter() - step_started
            next_id = int(logits.argmax())
            ids.append(next_id)
            generated.append(next_id)
        stats = {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.perf_counter() - started,
            "decode_tok_s": len(generated) / decode_seconds if decode_seconds > 0 else 0.0,
        }
        return generated, stats

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
    ) -> tuple[list[int], dict]:
        """Greedy generation by full recompute; returns (new_ids, stats)."""
        prompt_ids = [int(t) for t in prompt_ids]
        prompt_len = len(prompt_ids)
        ids = list(prompt_ids)
        generated: list[int] = []
        started = time.perf_counter()
        prefill_seconds = 0.0
        decode_seconds = 0.0
        for step in range(max_new_tokens):
            step_started = time.perf_counter()
            logits = self.forward(torch.tensor(ids, dtype=torch.long))
            elapsed = time.perf_counter() - step_started
            if step == 0:
                prefill_seconds = elapsed
            else:
                decode_seconds += elapsed
            next_id = int(logits[-1].argmax())
            ids.append(next_id)
            generated.append(next_id)
            if next_id in eos_token_ids:
                break
        stats = {
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.perf_counter() - started,
            "decode_tok_s": len(generated) / decode_seconds if decode_seconds > 0 else 0.0,
        }
        return generated, stats
