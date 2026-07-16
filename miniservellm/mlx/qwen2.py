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
    """Per-layer KV cache for single-request MLX decode.

    Stores K/V as a list of 4D arrays [batch, Nkv, capacity, Hd] — one per
    layer — instead of a single 5D array.  This keeps each ``slice_update``
    operating on a small per-layer array so the update cost does NOT scale
    with ``num_layers * capacity``.
    """

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
        shape = (batch_size, config.num_key_value_heads, max_context, config.head_dim)
        self.k_layers = [mx.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v_layers = [mx.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]

    def update(self, layer_idx: int, k_new: mx.array, v_new: mx.array) -> Tuple[mx.array, mx.array]:
        """Append [B, Hkv, T, D] KV and return the active continuous prefix."""
        length = k_new.shape[2]
        if self.offset + length > self.max_context:
            raise ValueError("MLX KV cache capacity exceeded")
        start = mx.array([0, 0, self.offset, 0], dtype=mx.uint32)
        self.k_layers[layer_idx] = mx.slice_update(self.k_layers[layer_idx], k_new, start, axes=(0, 1, 2, 3))
        self.v_layers[layer_idx] = mx.slice_update(self.v_layers[layer_idx], v_new, start, axes=(0, 1, 2, 3))
        return (
            self.k_layers[layer_idx][:, :, : self.offset + length, :],
            self.v_layers[layer_idx][:, :, : self.offset + length, :],
        )

    def advance(self, length: int) -> None:
        self.offset += length

    def clone(self) -> "MLXKVCache":
        """Create a cheap functional snapshot; later slice_update calls do not mutate this cache."""
        clone = object.__new__(MLXKVCache)
        clone.max_context = self.max_context
        clone.batch_size = self.batch_size
        clone.offset = self.offset
        clone.k_layers = list(self.k_layers)
        clone.v_layers = list(self.v_layers)
        return clone


class MLXQwen2:
    def __init__(self, config: ModelConfig, weights: MLXWeights):
        self.config = config
        self.weights = weights
        self.scale = config.head_dim ** -0.5
        self.q_dim = config.num_attention_heads * config.head_dim
        self.kv_dim = config.num_key_value_heads * config.head_dim
        self._compiled_decode: Optional[object] = None
        self._decode_attn_kernel: Optional[object] = None
        self._kernel_capacity: Optional[int] = None
        self._quant_bits: int = 0
        self._quant_group_size: int = 64

    def quantize_weights(self, bits: int = 4, group_size: int = 64) -> None:
        """Quantize all linear weight matrices in-place using MLX group quantization.

        After calling this, ``_linear`` automatically uses ``mx.quantized_matmul``
        for quantized weights (stored as ``(w_q, scales, biases)`` tuples).

        Embedding, LM head, and layernorm weights are left in fp16.
        """
        self._quant_bits = bits
        self._quant_group_size = group_size
        for layer in self.weights.layers:
            layer.qkv_proj = mx.quantize(layer.qkv_proj, group_size=group_size, bits=bits)
            layer.o_proj = mx.quantize(layer.o_proj, group_size=group_size, bits=bits)
            layer.gate_up_proj = mx.quantize(layer.gate_up_proj, group_size=group_size, bits=bits)
            layer.down_proj = mx.quantize(layer.down_proj, group_size=group_size, bits=bits)
        # Force recompile since weight representation changed
        self._compiled_decode = None

    def _linear(self, x: mx.array, weight, bias: Optional[mx.array] = None) -> mx.array:
        if isinstance(weight, (tuple, list)):
            w_q, scales, biases = weight
            y = mx.quantized_matmul(
                x, w_q, scales, biases,
                transpose=True,
                group_size=self._quant_group_size,
                bits=self._quant_bits,
            )
        else:
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
        # Decode (Q length=1): no causal mask needed — single query attends to all valid KV.
        # Prefill (Q length>1): causal mask required.
        attn_mask = None if length == 1 else "causal"
        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask=attn_mask)
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

    def _build_decode_attn_kernel(self, capacity: int) -> None:
        """Build a custom Metal kernel for batch=1 GQA decode attention.

        Uses online (Flash) softmax with simd_sum reduction.  Each KV head is
        handled by one SIMD group of 32 threads; the 7 query heads sharing a
        KV head reuse the same K/V load, cutting KV memory traffic 7×.

        ``capacity`` is baked in as a compile-time ``#define`` so it does not
        suffer fp16 precision loss.  ``position`` is passed at runtime as a
        two-element fp16 array (high/low split) to support values > 2048.
        """
        group_size = self.config.num_attention_heads // self.config.num_key_value_heads
        head_dim = self.config.head_dim
        num_kv_heads = self.config.num_key_value_heads
        assert head_dim == 64, "Custom decode kernel currently requires head_dim=64"
        assert head_dim % 32 == 0, "head_dim must be divisible by 32 (SIMD group size)"
        assert self.config.num_attention_heads % num_kv_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        header = (
            "using namespace metal;\n"
            f"#define NUM_HEADS {self.config.num_attention_heads}u\n"
            f"#define NUM_KV_HEADS {num_kv_heads}u\n"
            f"#define HEAD_DIM {head_dim}u\n"
            f"#define GROUP_SIZE {group_size}u\n"
            f"#define SCALE {self.scale:.10f}f\n"
            f"#define CAPACITY {capacity}u\n"
        )

        source = """
// gtid = global thread index; 32 threads per KV head (one SIMD group)
uint gtid = thread_position_in_grid.x;
uint kv_head = gtid / 32u;
uint lid = gtid % 32u;
uint d0 = lid * 2u;
uint d1 = lid * 2u + 1u;

// Reconstruct position from two fp16 halves (supports > 2048)
uint pos = uint(position[0]) * 2048u + uint(position[1]);
uint seq_len = pos + 1u;

// Load Q for GROUP_SIZE heads (2 dims per thread) into registers
float q_val[GROUP_SIZE][2];
for (uint g = 0; g < GROUP_SIZE; g++) {
    uint head = kv_head * GROUP_SIZE + g;
    q_val[g][0] = float(q[head * HEAD_DIM + d0]);
    q_val[g][1] = float(q[head * HEAD_DIM + d1]);
}

// Online softmax state (per Q head)
float running_max[GROUP_SIZE];
float running_sum[GROUP_SIZE];
float running_out[GROUP_SIZE][2];
for (uint g = 0; g < GROUP_SIZE; g++) {
    running_max[g] = -1e30f;
    running_sum[g] = 0.0f;
    running_out[g][0] = 0.0f;
    running_out[g][1] = 0.0f;
}

// Stream through KV positions
uint kv_base = kv_head * CAPACITY * HEAD_DIM;
for (uint s = 0; s < seq_len; s++) {
    uint kv_off = kv_base + s * HEAD_DIM;

    float k0 = float(k_cache[kv_off + d0]);
    float k1 = float(k_cache[kv_off + d1]);
    float v0 = float(v_cache[kv_off + d0]);
    float v1 = float(v_cache[kv_off + d1]);

    for (uint g = 0; g < GROUP_SIZE; g++) {
        float partial = q_val[g][0] * k0 + q_val[g][1] * k1;
        float score = simd_sum(partial) * SCALE;

        float new_max = max(running_max[g], score);
        float exp_diff = exp(running_max[g] - new_max);
        float exp_score = exp(score - new_max);
        running_sum[g] = running_sum[g] * exp_diff + exp_score;
        running_out[g][0] = running_out[g][0] * exp_diff + exp_score * v0;
        running_out[g][1] = running_out[g][1] * exp_diff + exp_score * v1;
        running_max[g] = new_max;
    }
}

// Write normalised output
for (uint g = 0; g < GROUP_SIZE; g++) {
    uint head = kv_head * GROUP_SIZE + g;
    float inv_sum = 1.0f / running_sum[g];
    out[head * HEAD_DIM + d0] = T(running_out[g][0] * inv_sum);
    out[head * HEAD_DIM + d1] = T(running_out[g][1] * inv_sum);
}
"""

        self._decode_attn_kernel = mx.fast.metal_kernel(
            name="decode_attn_gqa",
            input_names=["q", "k_cache", "v_cache", "position"],
            output_names=["out"],
            source=source,
            header=header,
        )
        self._kernel_capacity = capacity

    def _custom_decode_attention(
        self,
        q: mx.array,
        k_cache_layer: mx.array,
        v_cache_layer: mx.array,
        position: mx.array,
        capacity: int,
    ) -> mx.array:
        """Run the custom Metal decode-attention kernel.

        Args:
            q: [num_heads, head_dim] query (post-rope, squeezed).
            k_cache_layer: [num_kv_heads, capacity, head_dim].
            v_cache_layer: [num_kv_heads, capacity, head_dim].
            position: scalar mx.array — current decode offset.
            capacity: total KV cache capacity.

        Returns:
            [num_heads, head_dim] attention output.
        """
        # Lazy-build kernel with capacity baked in as #define
        if self._decode_attn_kernel is None or self._kernel_capacity != capacity:
            self._build_decode_attn_kernel(capacity)

        q_flat = q.reshape(-1)
        k_flat = k_cache_layer.reshape(-1)
        v_flat = v_cache_layer.reshape(-1)

        # Encode position as two fp16 halves to support values > 2048
        pos_val = int(position)
        position_encoded = mx.array(
            [float(pos_val // 2048), float(pos_val % 2048)],
            dtype=mx.float16,
        )

        grid_size = self.config.num_key_value_heads * 32
        result = self._decode_attn_kernel(
            inputs=[q_flat, k_flat, v_flat, position_encoded],
            template=[("T", mx.float16)],
            grid=(grid_size, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(self.config.num_attention_heads * self.config.head_dim,)],
            output_dtypes=[mx.float16],
        )
        return result[0].reshape(self.config.num_attention_heads, self.config.head_dim)

    def _decode_greedy_step(
        self,
        token_ids: mx.array,
        position: mx.array,
        k_layers: list,
        v_layers: list,
        use_custom_attn: bool = False,
    ) -> tuple[mx.array, list, list]:
        """Single batch=1 decode step returning (next_token, k_layers_new, v_layers_new)."""
        x = self.weights.embed_tokens[token_ids]
        capacity = k_layers[0].shape[2]
        if not use_custom_attn:
            valid_kv = mx.arange(capacity) <= position
            attention_mask = valid_kv.reshape(1, 1, 1, capacity)

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            qkv = self._linear(x_norm, layer.qkv_proj, layer.qkv_bias)
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

            q = q.reshape(1, 1, self.config.num_attention_heads, self.config.head_dim)
            k = k.reshape(1, 1, self.config.num_key_value_heads, self.config.head_dim)
            v = v.reshape(1, 1, self.config.num_key_value_heads, self.config.head_dim)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            q = mx.fast.rope(q, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)
            k = mx.fast.rope(k, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)

            start = mx.concatenate([
                mx.array([0, 0], dtype=mx.uint32),
                position.astype(mx.uint32).reshape(1),
                mx.array([0], dtype=mx.uint32),
            ])
            new_k_layers = list(k_layers)
            new_v_layers = list(v_layers)
            new_k_layers[layer_idx] = mx.slice_update(k_layers[layer_idx], k, start, axes=(0, 1, 2, 3))
            new_v_layers[layer_idx] = mx.slice_update(v_layers[layer_idx], v, start, axes=(0, 1, 2, 3))
            k_layers = new_k_layers
            v_layers = new_v_layers

            if use_custom_attn:
                q_2d = q.reshape(self.config.num_attention_heads, self.config.head_dim)
                k_layer = k_layers[layer_idx][0]
                v_layer = v_layers[layer_idx][0]
                attention = self._custom_decode_attention(
                    q_2d, k_layer, v_layer, position, capacity
                )
                attention = attention.reshape(1, 1, self.config.hidden_size)
            else:
                attention = mx.fast.scaled_dot_product_attention(
                    q, k_layers[layer_idx], v_layers[layer_idx], scale=self.scale, mask=attention_mask
                )
                attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(1, 1, self.config.hidden_size)
            x = residual + self._linear(attention, layer.o_proj)

            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        hidden = self._rms_norm(x, self.weights.final_norm)
        logits = self._linear(hidden, self.weights.lm_head)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        return next_token, k_layers, v_layers

    def get_compiled_decode_fn(self):
        """Return a cached mx.compile'd batch=1 decode step.

        The compiled function is created once and reused across calls.
        KV cache arrays are passed as arguments (not captured state),
        so the same compiled graph works for any cache of the same shape.
        """
        if self._compiled_decode is None:
            self._compiled_decode = mx.compile(self._decode_greedy_step)
        return self._compiled_decode

    def forward(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """Compatibility helper returning logits for the final input position only."""
        return self.project_last_hidden(self.forward_hidden(token_ids, cache))
