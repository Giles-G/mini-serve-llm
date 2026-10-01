"""Gemma4 MLX inference primitives.

This module is intentionally independent from the PyTorch Stage5 engine.  It
provides a dense, fixed-capacity batch=1 cache and an eager/compiled-friendly
forward for the Gemma4 text backbone.  Quantized linear layers are added only
after this FP16 path has parity coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import torch
from safetensors import safe_open

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.gemma4_config import build_gemma4_layer_specs
from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    TEXT_PREFIX,
    load_gemma4_config,
)


def _mx(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.float().numpy()).astype(mx.float16)


class _QuantizedRows:
    """Small-row lookup wrapper for the 9GB Gemma4 PLE embedding table."""

    def __init__(self, packed: mx.array, scales: mx.array, group_size: int = 64):
        self.packed = packed
        self.scales = scales
        self.group_size = group_size

    def __getitem__(self, ids: mx.array) -> mx.array:
        ids = ids.astype(mx.uint32).reshape(-1)
        packed = self.packed[ids]
        low = (packed.astype(mx.int32) & 15) - 8
        high = (packed.astype(mx.int32) >> 4) - 8
        q = mx.concatenate([low[..., None], high[..., None]], axis=-1)
        q = q.reshape(ids.shape[0], -1, self.group_size)
        scale = self.scales[ids].reshape(ids.shape[0], -1, 1)
        return (q.astype(mx.float16) * scale).reshape(ids.shape[0], -1)


def _quantize_rows(tensor: torch.Tensor, group_size: int = 64) -> _QuantizedRows:
    if tensor.shape[-1] % group_size:
        raise ValueError("PLE embedding width must be divisible by group_size")
    values = tensor.float().reshape(tensor.shape[0], -1, group_size)
    scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    quant = torch.round(values / scales).clamp(-8, 7).to(torch.int16) + 8
    quant = quant.reshape(tensor.shape[0], -1)
    packed = (quant[:, 0::2] | (quant[:, 1::2] << 4)).to(torch.uint8)
    return _QuantizedRows(
        mx.array(packed.numpy()),
        mx.array(scales.squeeze(-1).half().numpy()).astype(mx.float16),
    )


@dataclass
class Gemma4MLXLayer:
    q_proj: mx.array
    k_proj: mx.array
    v_proj: mx.array
    o_proj: mx.array
    q_norm: mx.array
    k_norm: mx.array
    input_layernorm: mx.array
    post_attention_layernorm: mx.array
    pre_feedforward_layernorm: mx.array
    post_feedforward_layernorm: mx.array
    post_per_layer_input_norm: mx.array
    gate_proj: mx.array
    up_proj: mx.array
    down_proj: mx.array
    per_layer_input_gate: mx.array
    per_layer_projection: mx.array
    layer_scalar: mx.array
    layer_spec: object


@dataclass
class Gemma4MLXWeights:
    embed_tokens: mx.array
    lm_head: mx.array
    embed_tokens_per_layer: mx.array
    per_layer_model_projection: mx.array
    per_layer_projection_norm: mx.array
    final_norm: mx.array
    layers: list[Gemma4MLXLayer]


class Gemma4MLXCache:
    """Preallocated cache grouped by the heterogeneous Gemma4 layer shape."""

    def __init__(self, config: ModelConfig, max_context: int, dtype=mx.float16):
        self.max_context = int(max_context)
        self.offset = 0
        self.k_layers: Dict[int, mx.array] = {}
        self.v_layers: Dict[int, mx.array] = {}
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        for spec in config.layer_specs:
            if spec.layer_idx >= first_shared:
                continue
            shape = (1, spec.num_key_value_heads, self.max_context, spec.head_dim)
            self.k_layers[spec.layer_idx] = mx.zeros(shape, dtype=dtype)
            self.v_layers[spec.layer_idx] = mx.zeros(shape, dtype=dtype)

    def update(self, layer_idx: int, k: mx.array, v: mx.array, start: int):
        length = int(k.shape[2])
        if start + length > self.max_context:
            raise ValueError("Gemma4 MLX KV cache capacity exceeded")
        begin = mx.array([0, 0, start, 0], dtype=mx.uint32)
        self.k_layers[layer_idx] = mx.slice_update(
            self.k_layers[layer_idx], k, begin, axes=(0, 1, 2, 3)
        )
        self.v_layers[layer_idx] = mx.slice_update(
            self.v_layers[layer_idx], v, begin, axes=(0, 1, 2, 3)
        )
        end = start + length
        return self.k_layers[layer_idx][:, :, :end, :], self.v_layers[layer_idx][:, :, :end, :]

    def reset(self) -> None:
        self.offset = 0


def load_gemma4_mlx_weights(
    model_dir: str | Path,
    model_config: ModelConfig,
) -> Gemma4MLXWeights:
    """Stream the text backbone from Gemma4 safetensors into MLX FP16 arrays."""

    path = Path(model_dir).expanduser()
    specs = build_gemma4_layer_specs(load_gemma4_config(path))
    handle = safe_open(str(path / "model.safetensors"), framework="pt", device="cpu")

    def read(name: str) -> mx.array:
        return _mx(handle.get_tensor(f"{TEXT_PREFIX}{name}"))

    def read_ple_embedding() -> _QuantizedRows:
        # This table is [262144, 8960] and is ~9.4GB in fp16, exceeding the
        # 8GB Metal buffer limit.  Keep it as packed row-wise INT4 and
        # dequantize only the token rows used by the current step.
        return _quantize_rows(
            handle.get_tensor(f"{TEXT_PREFIX}embed_tokens_per_layer.weight")
        )

    layers = []
    for spec in specs:
        i = spec.layer_idx
        layers.append(
            Gemma4MLXLayer(
                q_proj=read(f"layers.{i}.self_attn.q_proj.weight"),
                k_proj=read(f"layers.{i}.self_attn.k_proj.weight"),
                v_proj=read(f"layers.{i}.self_attn.v_proj.weight"),
                o_proj=read(f"layers.{i}.self_attn.o_proj.weight"),
                q_norm=read(f"layers.{i}.self_attn.q_norm.weight"),
                k_norm=read(f"layers.{i}.self_attn.k_norm.weight"),
                input_layernorm=read(f"layers.{i}.input_layernorm.weight"),
                post_attention_layernorm=read(f"layers.{i}.post_attention_layernorm.weight"),
                pre_feedforward_layernorm=read(f"layers.{i}.pre_feedforward_layernorm.weight"),
                post_feedforward_layernorm=read(f"layers.{i}.post_feedforward_layernorm.weight"),
                post_per_layer_input_norm=read(f"layers.{i}.post_per_layer_input_norm.weight"),
                gate_proj=read(f"layers.{i}.mlp.gate_proj.weight"),
                up_proj=read(f"layers.{i}.mlp.up_proj.weight"),
                down_proj=read(f"layers.{i}.mlp.down_proj.weight"),
                per_layer_input_gate=read(f"layers.{i}.per_layer_input_gate.weight"),
                per_layer_projection=read(f"layers.{i}.per_layer_projection.weight"),
                layer_scalar=read(f"layers.{i}.layer_scalar"),
                layer_spec=spec,
            )
        )
    embed_tokens = read("embed_tokens.weight")
    return Gemma4MLXWeights(
        embed_tokens=embed_tokens,
        lm_head=embed_tokens,
        embed_tokens_per_layer=read_ple_embedding(),
        per_layer_model_projection=read("per_layer_model_projection.weight"),
        per_layer_projection_norm=read("per_layer_projection_norm.weight"),
        final_norm=read("norm.weight"),
        layers=layers,
    )


class Gemma4MLX:
    """Gemma4 text forward with heterogeneous attention and KV sharing."""

    def __init__(self, config: ModelConfig, weights: Gemma4MLXWeights):
        self.config = config
        self.weights = weights
        self._rope_inv: dict[str, mx.array] = {}
        self.kv_source: dict[int, int] = {}
        self.quant_bits = 0
        self.quant_group_size = 64
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        prior = []
        for layer in weights.layers:
            spec = layer.layer_spec
            if spec.attention_type not in self._rope_inv:
                if spec.rope_type == "proportional":
                    rotated = spec.rotary_dim // 2
                    no_rot = spec.head_dim // 2 - rotated
                    inv = 1.0 / (
                        spec.rope_theta
                        ** (mx.arange(0, 2 * rotated, 2, dtype=mx.float32) / spec.head_dim)
                    )
                    if no_rot:
                        inv = mx.concatenate([inv, mx.zeros((no_rot,), dtype=mx.float32)])
                else:
                    inv = 1.0 / (
                        spec.rope_theta
                        ** (mx.arange(0, spec.head_dim, 2, dtype=mx.float32) / spec.head_dim)
                    )
                self._rope_inv[spec.attention_type] = inv
            if layer.layer_spec.layer_idx < first_shared:
                prior.append(layer)
            else:
                source = next(
                    item.layer_spec.layer_idx
                    for item in reversed(prior)
                    if item.layer_spec.attention_type == spec.attention_type
                )
                self.kv_source[spec.layer_idx] = source

    def quantize_weights(self, bits: int = 4, group_size: int = 64) -> None:
        """Quantize Gemma4 linear projections with MLX native group quantization."""
        if bits not in (4, 8):
            raise ValueError("MLX Gemma4 quantization supports 4 or 8 bits")
        self.quant_bits = bits
        self.quant_group_size = group_size
        # Keep input lookup in FP16, but quantize the tied output projection.
        self.weights.lm_head = mx.quantize(
            self.weights.embed_tokens,
            group_size=group_size,
            bits=bits,
        )
        names = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "per_layer_input_gate", "per_layer_projection",
        )
        for layer in self.weights.layers:
            for name in names:
                setattr(
                    layer,
                    name,
                    mx.quantize(
                        getattr(layer, name),
                        group_size=group_size,
                        bits=bits,
                    ),
                )

    def _linear(self, x: mx.array, weight: mx.array) -> mx.array:
        if isinstance(weight, (tuple, list)):
            return mx.quantized_matmul(
                x,
                weight[0],
                weight[1],
                weight[2],
                transpose=True,
                group_size=self.quant_group_size,
                bits=self.quant_bits,
            )
        return x @ weight.T

    def _norm(self, x: mx.array, weight: mx.array) -> mx.array:
        if weight is None:
            return x * mx.rsqrt(
                mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
                + self.config.rms_norm_eps
            ).astype(x.dtype)
        return mx.fast.rms_norm(x, weight, self.config.rms_norm_eps)

    @staticmethod
    def _gelu_tanh(x: mx.array) -> mx.array:
        return 0.5 * x * (
            1.0
            + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
        )

    @staticmethod
    def _gqa_attention(q, k, v, mask, num_kv_heads):
        """GQA attention without materializing repeated K/V heads."""
        batch, num_q_heads, query_len, head_dim = q.shape
        repeat = num_q_heads // num_kv_heads
        if repeat == 1:
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
            probs = mx.softmax((scores + mask).astype(mx.float32), axis=-1).astype(v.dtype)
            return mx.matmul(probs, v)
        q_grouped = q.reshape(batch, num_kv_heads, repeat, query_len, head_dim)
        k_grouped = mx.transpose(k[:, :, None, :, :], (0, 1, 2, 4, 3))
        scores = mx.matmul(q_grouped, k_grouped)
        probs = mx.softmax((scores + mask).astype(mx.float32), axis=-1).astype(v.dtype)
        out = mx.matmul(probs, v[:, :, None, :, :])
        return out.reshape(batch, num_q_heads, query_len, head_dim)

    def _rope(self, x: mx.array, layer_idx: int, positions: mx.array) -> mx.array:
        spec = self.weights.layers[layer_idx].layer_spec
        inv = self._rope_inv[spec.attention_type]
        angles = positions[:, None] * inv[None, :]
        emb = mx.concatenate([angles, angles], axis=-1)
        # x is [batch, time, heads, dim]; position embeddings are [time, dim].
        cos = mx.cos(emb)[None, :, None, :]
        sin = mx.sin(emb)[None, :, None, :]
        half = x.shape[-1] // 2
        rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rotated * sin

    def _ple(self, ids: mx.array, x: mx.array) -> mx.array:
        dim = self.config.hidden_size_per_layer_input
        if not dim:
            return None
        identity = self.weights.embed_tokens_per_layer[ids]
        identity = identity.reshape(ids.shape[0], ids.shape[1], self.config.num_hidden_layers, dim)
        context = self._linear(x, self.weights.per_layer_model_projection)
        context = context.reshape(ids.shape[0], ids.shape[1], self.config.num_hidden_layers, dim)
        context = context * (self.config.hidden_size ** -0.5)
        context = self._norm(context, self.weights.per_layer_projection_norm)
        return (identity * (dim ** 0.5) + context) * (2.0 ** -0.5)

    def _attention(self, h, layer, cache, start):
        spec = layer.layer_spec
        b, t, _ = h.shape
        q = self._linear(h, layer.q_proj).reshape(b, t, spec.num_attention_heads, spec.head_dim)
        q = self._norm(q, layer.q_norm)
        q = self._rope(q, spec.layer_idx, mx.arange(start, start + t, dtype=mx.float32))
        q = mx.transpose(q, (0, 2, 1, 3))
        source = self.kv_source.get(spec.layer_idx)
        if source is None:
            k = self._linear(h, layer.k_proj).reshape(b, t, spec.num_key_value_heads, spec.head_dim)
            k = self._norm(k, layer.k_norm)
            k = self._rope(k, spec.layer_idx, mx.arange(start, start + t, dtype=mx.float32))
            v = self._linear(h, layer.v_proj).reshape(b, t, spec.num_key_value_heads, spec.head_dim)
            v = self._norm(v, None)
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            k_all, v_all = cache.update(spec.layer_idx, k, v, start)
        else:
            k_all = cache.k_layers[source][:, :, : start + t, :]
            v_all = cache.v_layers[source][:, :, : start + t, :]
        key_start = 0
        if spec.attention_type == "sliding_attention" and spec.sliding_window:
            # Earliest query needs its entire window, not just the last
            # query's window (important for multi-token prefill).
            key_start = max(0, start - spec.sliding_window + 1)
            k_all = k_all[:, :, key_start:, :]
            v_all = v_all[:, :, key_start:, :]
        q_abs = mx.arange(start, start + t)[:, None]
        k_abs = mx.arange(key_start, start + t)[None, :]
        allowed = k_abs <= q_abs
        if spec.attention_type == "sliding_attention" and spec.sliding_window:
            allowed = allowed & ((q_abs - k_abs) < spec.sliding_window)
        mask = mx.where(allowed, mx.array(0.0), mx.array(float("-inf")))
        repeat = spec.num_attention_heads // spec.num_key_value_heads
        if repeat > 1:
            k_all = mx.repeat(k_all, repeat, axis=1)
            v_all = mx.repeat(v_all, repeat, axis=1)
        out = mx.matmul(q, mx.transpose(k_all, (0, 1, 3, 2)))
        out = out + mask[None, None, :, :]
        probs = mx.softmax(out.astype(mx.float32), axis=-1).astype(v_all.dtype)
        out = mx.matmul(probs, v_all)
        return mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, -1)

    def forward(self, token_ids: mx.array, cache: Gemma4MLXCache) -> mx.array:
        """Return logits [B, T, vocab] and update cache."""
        b, t = token_ids.shape
        x = self.weights.embed_tokens[token_ids] * (self.config.hidden_size ** 0.5)
        ple = self._ple(token_ids, x)
        start = cache.offset
        for layer in self.weights.layers:
            residual = x
            h = self._norm(x, layer.input_layernorm)
            attn = self._attention(h, layer, cache, start)
            h = self._linear(attn, layer.o_proj)
            x = residual + self._norm(h, layer.post_attention_layernorm)
            residual = x
            h = self._norm(x, layer.pre_feedforward_layernorm)
            gate = self._gelu_tanh(self._linear(h, layer.gate_proj))
            h = self._linear(gate * self._linear(h, layer.up_proj), layer.down_proj)
            x = residual + self._norm(h, layer.post_feedforward_layernorm)
            if ple is not None:
                residual = x
                h = self._gelu_tanh(self._linear(x, layer.per_layer_input_gate))
                h = h * ple[:, :, layer.layer_spec.layer_idx, :]
                h = self._linear(h, layer.per_layer_projection)
                x = residual + self._norm(h, layer.post_per_layer_input_norm)
            x = x * layer.layer_scalar
        cache.offset += t
        x = self._norm(x, self.weights.final_norm)
        logits = self._linear(x, self.weights.lm_head).astype(mx.float32)
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = softcap * mx.tanh(logits / softcap)
        return logits

    def decode(self, token_id: mx.array, cache: Gemma4MLXCache) -> mx.array:
        return self.forward(token_id.reshape(1, 1), cache)[:, -1, :]

    def _compiled_decode_step(
        self,
        token_id: mx.array,
        position: mx.array,
        k_layers: list[mx.array],
        v_layers: list[mx.array],
    ):
        """Functional fixed-capacity decode graph.

        The cache arrays are explicit graph inputs/outputs.  Keeping the full
        capacity in the attention mask avoids dynamic slicing and lets MLX
        compile one stable batch=1 graph per capacity/quantization setting.
        """
        x = self.weights.embed_tokens[token_id.reshape(1, 1)]
        x = x * (self.config.hidden_size ** 0.5)
        ple_ids = token_id.reshape(1, 1)
        ple = self._ple(ple_ids, x)
        capacity = k_layers[0].shape[2]
        key_pos = mx.arange(capacity, dtype=mx.int32).reshape(1, 1, 1, capacity)
        position_i = position.astype(mx.int32).reshape(1, 1, 1, 1)
        common_allowed = key_pos <= position_i
        new_k_layers = list(k_layers)
        new_v_layers = list(v_layers)

        for layer in self.weights.layers:
            spec = layer.layer_spec
            residual = x
            h = self._norm(x, layer.input_layernorm)
            q = self._linear(h, layer.q_proj).reshape(
                1, 1, spec.num_attention_heads, spec.head_dim
            )
            q = self._norm(q, layer.q_norm)
            q = self._rope(q, spec.layer_idx, position.astype(mx.float32).reshape(1))
            q = mx.transpose(q, (0, 2, 1, 3))

            source = self.kv_source.get(spec.layer_idx)
            if source is None:
                k = self._linear(h, layer.k_proj).reshape(
                    1, 1, spec.num_key_value_heads, spec.head_dim
                )
                k = self._norm(k, layer.k_norm)
                k = self._rope(k, spec.layer_idx, position.astype(mx.float32).reshape(1))
                k = mx.transpose(k, (0, 2, 1, 3))
                v = self._linear(h, layer.v_proj).reshape(
                    1, 1, spec.num_key_value_heads, spec.head_dim
                )
                v = self._norm(v, None)
                v = mx.transpose(v, (0, 2, 1, 3))
                start = mx.array([0, 0, 0, 0], dtype=mx.uint32)
                # Dynamic start is represented by a tensor in the compiled
                # graph; the third coordinate is the current token position.
                start = mx.stack(
                    [mx.array(0, dtype=mx.uint32), mx.array(0, dtype=mx.uint32),
                     position.astype(mx.uint32), mx.array(0, dtype=mx.uint32)]
                )
                new_k_layers[spec.layer_idx] = mx.slice_update(
                    k_layers[spec.layer_idx], k, start, axes=(0, 1, 2, 3)
                )
                new_v_layers[spec.layer_idx] = mx.slice_update(
                    v_layers[spec.layer_idx], v, start, axes=(0, 1, 2, 3)
                )
                k_all = new_k_layers[spec.layer_idx]
                v_all = new_v_layers[spec.layer_idx]
            else:
                k_all = new_k_layers[source]
                v_all = new_v_layers[source]

            if (
                spec.attention_type == "sliding_attention"
                and spec.sliding_window
                and capacity > spec.sliding_window
            ):
                # Once the context exceeds the window, gather only the fixed
                # sliding tail.  For shorter contexts retain the original
                # full-capacity masked path to avoid dynamic small-context
                # overhead.
                window = spec.sliding_window
                start_pos = position.astype(mx.int32) - window + 1
                indices = start_pos + mx.arange(window, dtype=mx.int32)
                indices = mx.maximum(indices, mx.array(0, dtype=mx.int32))
                k_all = mx.take(k_all, indices, axis=2)
                v_all = mx.take(v_all, indices, axis=2)
                valid = mx.arange(window, dtype=mx.int32) <= position
                mask = mx.where(
                    valid.reshape(1, 1, 1, window),
                    mx.array(0.0),
                    mx.array(float("-inf")),
                )
            else:
                allowed = common_allowed
                if spec.attention_type == "sliding_attention" and spec.sliding_window:
                    allowed = allowed & (
                        key_pos >= position_i - spec.sliding_window + 1
                    )
                mask = mx.where(allowed, mx.array(0.0), mx.array(float("-inf")))
            repeat = spec.num_attention_heads // spec.num_key_value_heads
            if repeat > 1:
                k_all = mx.repeat(k_all, repeat, axis=1)
                v_all = mx.repeat(v_all, repeat, axis=1)
            scores = mx.matmul(q, mx.transpose(k_all, (0, 1, 3, 2)))
            attn = mx.softmax((scores + mask).astype(mx.float32), axis=-1).astype(v_all.dtype)
            attn = mx.matmul(attn, v_all)
            attn = mx.transpose(attn, (0, 2, 1, 3)).reshape(
                1, 1, spec.num_attention_heads * spec.head_dim
            )
            h = self._linear(attn, layer.o_proj)
            x = residual + self._norm(h, layer.post_attention_layernorm)

            residual = x
            h = self._norm(x, layer.pre_feedforward_layernorm)
            gate = self._gelu_tanh(self._linear(h, layer.gate_proj))
            h = self._linear(gate * self._linear(h, layer.up_proj), layer.down_proj)
            x = residual + self._norm(h, layer.post_feedforward_layernorm)
            if ple is not None:
                residual = x
                h = self._gelu_tanh(self._linear(x, layer.per_layer_input_gate))
                h = h * ple[:, :, spec.layer_idx, :]
                h = self._linear(h, layer.per_layer_projection)
                x = residual + self._norm(h, layer.post_per_layer_input_norm)
            x = x * layer.layer_scalar

        logits = self._linear(self._norm(x, self.weights.final_norm), self.weights.lm_head)
        logits = logits.astype(mx.float32)
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = softcap * mx.tanh(logits / softcap)
        # Decode only needs argmax/top-k.  Keep the compiled graph output in
        # fp16 after numerically stable softcap computation to halve the
        # 262k-vocabulary result bandwidth.
        return logits[:, -1, :].astype(mx.float16), new_k_layers, new_v_layers

    def get_compiled_decode(self):
        """Return a compiled fixed-capacity decode function."""
        return mx.compile(self._compiled_decode_step)

    def _compiled_decode_step_greedy(
        self,
        token_id: mx.array,
        position: mx.array,
        k_layers: list[mx.array],
        v_layers: list[mx.array],
    ):
        """Compiled decode that returns the next token, not the full logits."""
        logits, new_k_layers, new_v_layers = self._compiled_decode_step(
            token_id, position, k_layers, v_layers
        )
        return mx.argmax(logits, axis=-1).astype(mx.uint32), new_k_layers, new_v_layers

    def get_compiled_decode_greedy(self):
        """Return a compiled decode graph with argmax fused into its output."""
        return mx.compile(self._compiled_decode_step_greedy)


def build_gemma4_mlx(
    model_dir: str | Path,
    model_config: ModelConfig,
) -> Gemma4MLX:
    return Gemma4MLX(model_config, load_gemma4_mlx_weights(model_dir, model_config))
