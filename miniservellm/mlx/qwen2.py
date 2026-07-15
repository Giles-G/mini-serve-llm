"""Minimal MLX Qwen2 FP16 implementation for Apple Silicon batch=1 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from miniservellm.config import ModelConfig


@dataclass
class MLXLayerWeights:
    qkv_proj: mx.array
    qkv_bias: Optional[mx.array]
    o_proj: mx.array
    gate_up_proj: mx.array
    down_proj: mx.array
    input_layernorm: mx.array
    post_attention_layernorm: mx.array


@dataclass
class MLXWeights:
    embed_tokens: mx.array
    layers: List[MLXLayerWeights]
    final_norm: mx.array
    lm_head: mx.array


class MLXKVCache:
    """Continuous per-layer KV cache for single-request MLX decode."""

    def __init__(
        self,
        config: ModelConfig,
        max_context: int,
        batch_size: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        self.max_context = max_context
        self.batch_size = batch_size
        self.offset = 0
        shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context,
            config.head_dim,
        )
        self.k = mx.zeros(shape, dtype=dtype)
        self.v = mx.zeros(shape, dtype=dtype)

    def update(self, layer_idx: int, k_new: mx.array, v_new: mx.array) -> Tuple[mx.array, mx.array]:
        """Append [1, Hkv, T, D] KV and return the active continuous prefix."""
        length = k_new.shape[2]
        if self.offset + length > self.max_context:
            raise ValueError("MLX KV cache capacity exceeded")
        start = mx.array([layer_idx, 0, 0, self.offset, 0], dtype=mx.uint32)
        self.k = mx.slice_update(self.k, k_new[None], start, axes=(0, 1, 2, 3, 4))
        self.v = mx.slice_update(self.v, v_new[None], start, axes=(0, 1, 2, 3, 4))
        return (
            self.k[layer_idx, :, :, : self.offset + length, :],
            self.v[layer_idx, :, :, : self.offset + length, :],
        )

    def advance(self, length: int) -> None:
        self.offset += length

    def clone(self) -> "MLXKVCache":
        """Create a cheap functional snapshot; later slice_update calls do not mutate this cache."""
        clone = object.__new__(MLXKVCache)
        clone.max_context = self.max_context
        clone.batch_size = self.batch_size
        clone.offset = self.offset
        clone.k = self.k
        clone.v = self.v
        return clone


class MLXQwen2:
    def __init__(self, config: ModelConfig, weights: MLXWeights):
        self.config = config
        self.weights = weights
        self.scale = config.head_dim ** -0.5
        self.q_dim = config.num_attention_heads * config.head_dim
        self.kv_dim = config.num_key_value_heads * config.head_dim

    def _linear(self, x: mx.array, weight: mx.array, bias: Optional[mx.array] = None) -> mx.array:
        y = mx.matmul(x, weight.T)
        return y if bias is None else y + bias

    def _rms_norm(self, x: mx.array, weight: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight, self.config.rms_norm_eps)

    def _attention(
        self,
        x: mx.array,
        layer_idx: int,
        layer: MLXLayerWeights,
        cache: MLXKVCache,
        offset: int,
    ) -> mx.array:
        batch, length, _ = x.shape
        qkv = self._linear(x, layer.qkv_proj, layer.qkv_bias)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        q = q.reshape(batch, length, self.config.num_attention_heads, self.config.head_dim)
        k = k.reshape(batch, length, self.config.num_key_value_heads, self.config.head_dim)
        v = v.reshape(batch, length, self.config.num_key_value_heads, self.config.head_dim)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q = mx.fast.rope(q, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=offset)

        k_all, v_all = cache.update(layer_idx, k, v)
        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask="causal")
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, length, self.config.hidden_size)
        return self._linear(out, layer.o_proj)

    def forward_hidden(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """Forward [B, T] tokens and return final hidden states after KV update."""
        x = self.weights.embed_tokens[token_ids]
        offset = cache.offset
        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            x = residual + self._attention(x_norm, layer_idx, layer, cache, offset)

            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        cache.advance(token_ids.shape[1])
        return self._rms_norm(x, self.weights.final_norm)

    def project_last_hidden(self, hidden_states: mx.array) -> mx.array:
        """Project only the final token state to vocabulary logits."""
        return self._linear(hidden_states[:, -1:, :], self.weights.lm_head)

    def forward(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """Compatibility helper returning logits for the final input position only."""
        return self.project_last_hidden(self.forward_hidden(token_ids, cache))
