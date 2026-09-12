"""Minimal MLX Qwen2 FP16 implementation for Apple Silicon batch=1 inference.

面向 Apple Silicon 的极简 Qwen2 推理实现（MLX 后端，batch=1 为主）：
- 权重布局：QKV 与 Gate/Up 纵向融合，减少 matmul 次数；支持分组量化；
- KV Cache：逐层存储 + 每行独立写游标，兼容等长静态 batch 与动态 slot；
- 三条解码路径：
    eager forward        -> prefill / 通用路径（forward_hidden）
    mx.compile 静态图     -> 等长 batch 共享游标（_static_decode_step*）
                            各行独立游标（_dynamic_decode_step）
    自定义 Metal kernel   -> batch=1 GQA decode attention（_decode_greedy_step）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from miniservellm.config import ModelConfig


@dataclass
class MLXLayerWeights:
    """单个 Transformer 层的权重（QKV 与 Gate/Up 纵向融合，减少 matmul 次数）。"""

    qkv_proj: mx.array                  # [hidden, q_dim + 2*kv_dim]，融合的 QKV 投影
    qkv_bias: Optional[mx.array]        # QKV 偏置（Qwen2 特有，可为 None）
    o_proj: mx.array                    # [hidden, hidden]，注意力输出投影
    gate_up_proj: mx.array              # [hidden, 2*inter]，融合的 Gate/Up 投影（SwiGLU）
    down_proj: mx.array                 # [inter, hidden]，FFN 下投影
    input_layernorm: mx.array           # 注意力前 RMSNorm 权重
    post_attention_layernorm: mx.array  # FFN 前 RMSNorm 权重


@dataclass
class MLXWeights:
    """整模型权重：embedding + 各层权重 + 最终 RMSNorm + LM head。"""

    embed_tokens: mx.array              # [vocab, hidden]，词嵌入表
    layers: List[MLXLayerWeights]
    final_norm: mx.array                # 输出前最后一层 RMSNorm 权重
    lm_head: mx.array                   # [hidden, vocab]，输出头（tied embeddings，与 embedding 共享同一数组）


class MLXKVCache:
    """MLX 解码用逐层 KV Cache。

    K/V 按层存储为 [batch, Nkv, capacity, Hd] 的 4D 数组列表（每层一份），
    而不是单个 5D 数组——这样每次 ``slice_update`` 只作用于单层小数组，
    更新开销不会随 ``num_layers * capacity`` 放大。

    每个 batch 行维护独立的写游标 ``slot_offsets``：
    - 等长静态 batch 各行游标相同，走共享游标的 ``update`` 路径；
    - 连续服务（各行进度不同）走 ``update_decode_rows`` 逐行散射。
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
        # 每个 batch 行拥有独立的 KV 写游标；等长静态 batch 各行相同，连续服务则不同
        self.slot_offsets = [0] * batch_size
        # 每层预分配整段容量的 K/V 缓冲，decode 时原地 slice_update，无需扩容
        shape = (batch_size, config.num_key_value_heads, max_context, config.head_dim)
        self.k_layers = [mx.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v_layers = [mx.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]

    @property
    def offset(self) -> int:
        """共享游标的兼容读取器：仅当各行游标一致时可用（旧等长调用方）。"""
        if len(set(self.slot_offsets)) != 1:
            raise ValueError("MLXKVCache has per-slot offsets; use slot_offsets instead of offset")
        return self.slot_offsets[0]

    def set_slot_offsets(self, offsets: List[int]) -> None:
        """设置各 batch 行的独立写游标（连续服务路径用）。"""
        if len(offsets) != self.batch_size:
            raise ValueError("offset count must match batch_size")
        if any(offset < 0 or offset > self.max_context for offset in offsets):
            raise ValueError("slot offset is outside cache capacity")
        self.slot_offsets = list(offsets)

    def update(self, layer_idx: int, k_new: mx.array, v_new: mx.array) -> Tuple[mx.array, mx.array]:
        """在共享游标处为所有行追加等长的 K/V（prefill 与等长 batch decode 用）。

        返回截至 (游标 + length) 的有效 K/V 切片 [batch, Nkv, seq, Hd]，
        供注意力直接使用；容量不足时抛错。
        """
        if len(set(self.slot_offsets)) != 1:
            raise ValueError("update requires equal slot offsets; use update_decode_rows")
        length = k_new.shape[2]
        if self.offset + length > self.max_context:
            raise ValueError("MLX KV cache capacity exceeded")
        start = mx.array([0, 0, self.offset, 0], dtype=mx.uint32)
        self.k_layers[layer_idx] = mx.slice_update(self.k_layers[layer_idx], k_new, start, axes=(0, 1, 2, 3))
        self.v_layers[layer_idx] = mx.slice_update(self.v_layers[layer_idx], v_new, start, axes=(0, 1, 2, 3))
        end = self.offset + length
        return self.k_layers[layer_idx][:, :, :end, :], self.v_layers[layer_idx][:, :, :end, :]

    def update_decode_rows(
        self,
        layer_idx: int,
        k_new: mx.array,
        v_new: mx.array,
        active_rows: List[bool],
    ) -> Tuple[mx.array, mx.array]:
        """按各行独立游标，为每个活跃行散射写入一个 KV token（decode 单步）。

        每 K/V 层各用一次 ``put_along_axis`` 完成整 batch 写入，取代旧的
        逐行 ``slice_update`` + batch 轴拼接。非活跃行把该位置的旧值原样
        写回（仅占位，不改变 cache 内容）。
        """
        if k_new.shape[2] != 1 or v_new.shape[2] != 1:
            raise ValueError("update_decode_rows requires exactly one token per row")
        if len(active_rows) != self.batch_size:
            raise ValueError("active_rows count must match batch_size")
        if any(active and offset >= self.max_context for offset, active in zip(self.slot_offsets, active_rows)):
            raise ValueError("MLX KV cache capacity exceeded")

        k_layer = self.k_layers[layer_idx]
        v_layer = self.v_layers[layer_idx]
        # 已结束的行游标可能停在 capacity 上，钳到范围内保证散射索引合法
        write_offsets = [min(offset, self.max_context - 1) for offset in self.slot_offsets]
        offsets = mx.array(write_offsets, dtype=mx.uint32).reshape(self.batch_size, 1, 1, 1)
        indices = mx.broadcast_to(offsets, (self.batch_size, k_layer.shape[1], 1, k_layer.shape[3]))
        active_mask = mx.array(active_rows).reshape(self.batch_size, 1, 1, 1)
        k_existing = mx.take_along_axis(k_layer, indices, axis=2)
        v_existing = mx.take_along_axis(v_layer, indices, axis=2)
        k_updates = mx.where(active_mask, k_new, k_existing)
        v_updates = mx.where(active_mask, v_new, v_existing)
        k_layer = mx.put_along_axis(k_layer, indices, k_updates, axis=2)
        v_layer = mx.put_along_axis(v_layer, indices, v_updates, axis=2)
        self.k_layers[layer_idx] = k_layer
        self.v_layers[layer_idx] = v_layer
        return k_layer, v_layer

    def advance(self, length: int) -> None:
        """所有行游标统一前进 length 步（prefill / 等长 batch decode 用）。"""
        self.slot_offsets = [offset + length for offset in self.slot_offsets]

    def advance_slots(self, active_rows: List[bool]) -> None:
        """仅活跃行的游标前进 1 步（动态 decode 用）。"""
        if len(active_rows) != self.batch_size:
            raise ValueError("active_rows count must match batch_size")
        self.slot_offsets = [offset + int(active) for offset, active in zip(self.slot_offsets, active_rows)]

    def clone(self) -> "MLXKVCache":
        """创建廉价的函数式快照：只复制列表引用，后续 slice_update 不会改到本 cache。"""
        clone = object.__new__(MLXKVCache)
        clone.max_context = self.max_context
        clone.batch_size = self.batch_size
        clone.slot_offsets = list(self.slot_offsets)
        clone.k_layers = list(self.k_layers)
        clone.v_layers = list(self.v_layers)
        return clone


class MLXQwen2:
    """极简 Qwen2 前向实现：eager / mx.compile / 自定义 Metal kernel 三条解码路径。"""

    def __init__(self, config: ModelConfig, weights: MLXWeights):
        self.config = config
        self.weights = weights
        self.scale = config.head_dim ** -0.5  # 注意力缩放因子 1/sqrt(head_dim)
        self.q_dim = config.num_attention_heads * config.head_dim
        self.kv_dim = config.num_key_value_heads * config.head_dim
        # ---- 编译图缓存：按 (量化位数, batch, capacity, dtype) 键控，避免重复编译 ----
        self._compiled_decode: Optional[object] = None
        self._compiled_static_decodes: dict[tuple[int, int, int, str], object] = {}
        self._compiled_static_decode_hits = 0
        self._compiled_static_decode_misses = 0
        self._compiled_static_decode_greedy: dict[tuple[int, int, int, str], object] = {}
        self._compiled_static_decode_greedy_hits = 0
        self._compiled_static_decode_greedy_misses = 0
        self._compiled_dynamic_decodes: dict[tuple[int, int, int, str], object] = {}
        self._compiled_dynamic_decode_hits = 0
        self._compiled_dynamic_decode_misses = 0
        self._decode_attn_kernel: Optional[object] = None
        self._kernel_capacity: Optional[int] = None
        self._quant_bits: int = 0
        self._quant_group_size: int = 64

    def quantize_weights(self, bits: int = 4, group_size: int = 64) -> None:
        """就地分组量化所有线性层权重（MLX group quantization）。

        量化后权重变为 ``(w_q, scales, biases)`` 三元组，``_linear`` 会自动
        切换到 ``mx.quantized_matmul``。embedding、LM head 和 layernorm
        保留 fp16（对精度敏感且量化收益小）。
        """
        self._quant_bits = bits
        self._quant_group_size = group_size
        for layer in self.weights.layers:
            layer.qkv_proj = mx.quantize(layer.qkv_proj, group_size=group_size, bits=bits)
            layer.o_proj = mx.quantize(layer.o_proj, group_size=group_size, bits=bits)
            layer.gate_up_proj = mx.quantize(layer.gate_up_proj, group_size=group_size, bits=bits)
            layer.down_proj = mx.quantize(layer.down_proj, group_size=group_size, bits=bits)
        # 权重表示变了，丢弃所有已编译解码图并重置统计
        self._compiled_decode = None
        self._compiled_static_decodes.clear()
        self._compiled_static_decode_hits = 0
        self._compiled_static_decode_misses = 0
        self._compiled_static_decode_greedy.clear()
        self._compiled_static_decode_greedy_hits = 0
        self._compiled_static_decode_greedy_misses = 0
        self._compiled_dynamic_decodes.clear()
        self._compiled_dynamic_decode_hits = 0
        self._compiled_dynamic_decode_misses = 0

    def _linear(self, x: mx.array, weight, bias: Optional[mx.array] = None) -> mx.array:
        """线性层：weight 为量化三元组时走 quantized_matmul，否则普通 matmul。"""
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
        """RMSNorm（走 mx.fast 融合算子）。"""
        return mx.fast.rms_norm(x, weight, self.config.rms_norm_eps)

    def _apply_rope_per_row(self, x: mx.array, positions: mx.array) -> mx.array:
        """纯张量实现的逐行 RoPE（各行位置偏移可以不同）。

        取代逐行调用 ``mx.fast.rope`` 的 Python 循环——后者破坏算子融合，
        且当 offset 是运行时值时无法被 ``mx.compile`` 追踪。

        Args:
            x: [batch, nheads, seqlen, head_dim]
            positions: [batch] float32 — 各行的 decode 位置偏移。
        """
        half = self.config.head_dim // 2
        inv_freq = self.config.rope_theta ** (
            mx.arange(0, self.config.head_dim, 2, dtype=mx.float32) / -self.config.head_dim
        )  # [half]
        angles = positions[:, None] * inv_freq[None, :]  # [batch, half]
        cos = mx.cos(angles)[:, None, None, :]  # [batch, 1, 1, half]
        sin = mx.sin(angles)[:, None, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def _attention(
        self,
        x: mx.array,
        layer_idx: int,
        layer: MLXLayerWeights,
        cache: MLXKVCache,
        offset: int,
    ) -> mx.array:
        """普通注意力路径（eager prefill / 等长 batch decode，共享游标写 KV）。

        Args:
            offset: 当前 KV 写入的起始位置（prefill 为 0，decode 为游标值）。
        """
        batch, length, _ = x.shape
        # 融合 QKV 一次投影，再按维度切开
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
        # decode（Q 长度=1）：单 query 对所有有效 KV，无需 causal mask
        # prefill（Q 长度>1）：必须用 causal mask，防止看到未来 token
        attn_mask = None if length == 1 else "causal"
        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask=attn_mask)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, length, self.config.hidden_size)
        return self._linear(out, layer.o_proj)

    def forward_hidden(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """eager 前向：输入 [B, T] token，更新 KV 后返回最终隐藏状态。

        标准 prefill 路径；每层结构为
        x += Attn(RMSNorm(x)) ; x += FFN(RMSNorm(x))，末尾做 final_norm。
        """
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
        output = self._rms_norm(x, self.weights.final_norm)
        return output

    def forward_hidden_dynamic_decode(
        self,
        token_ids: mx.array,
        cache: MLXKVCache,
        active_rows: List[bool],
    ) -> mx.array:
        """动态 decode（eager 回退路径）：每个活跃行各解一个 token，游标独立。

        逐行 RoPE 偏移 + 批量 KV 散射；对应的编译版固定形状图
        （``_dynamic_decode_step``）把 cache 数组与 slot 元数据作为显式图输入。
        """
        if token_ids.shape != (cache.batch_size, 1):
            raise ValueError("token_ids must have shape [batch_size, 1]")
        if len(active_rows) != cache.batch_size:
            raise ValueError("active_rows count must match batch_size")

        x = self.weights.embed_tokens[token_ids]
        # 构造 [B, 1, 1, capacity] 的注意力掩码：
        # - 活跃行：只允许看到位置 <= 自身游标的 KV；
        # - 非活跃行：输入是占位 token，掩码只留位置 0，保证数值合法
        #   （避免全掩码行产生 NaN），其输出后续会被丢弃。
        capacity = cache.max_context
        positions = mx.arange(capacity)[None, None, None, :]
        offsets = mx.array(cache.slot_offsets, dtype=mx.int32)[:, None, None, None]
        valid_kv = positions <= offsets
        inactive_kv = mx.arange(capacity)[None, None, None, :] == 0
        active_mask = mx.array(active_rows).reshape(cache.batch_size, 1, 1, 1)
        attention_mask = mx.where(active_mask, valid_kv, inactive_kv)

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            qkv = self._linear(x_norm, layer.qkv_proj, layer.qkv_bias)
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
            q = mx.transpose(q.reshape(cache.batch_size, 1, self.config.num_attention_heads, self.config.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(cache.batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(cache.batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))

            # Per-row RoPE via tensor ops (no Python loop)
            positions = mx.array(cache.slot_offsets, dtype=mx.float32)
            q = self._apply_rope_per_row(q, positions)
            k = self._apply_rope_per_row(k, positions)
            k_all, v_all = cache.update_decode_rows(layer_idx, k, v, active_rows)
            attention = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask=attention_mask)
            attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(cache.batch_size, 1, self.config.hidden_size)
            x = residual + self._linear(attention, layer.o_proj)
            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        cache.advance_slots(active_rows)
        return self._rms_norm(x, self.weights.final_norm)

    def project_last_hidden(self, hidden_states: mx.array) -> mx.array:
        """仅把最后一个 token 的隐藏状态投影到词表 logits（decode 单步只差一个位置）。"""
        return self._linear(hidden_states[:, -1:, :], self.weights.lm_head)

    def _static_decode_step(
        self,
        token_ids: mx.array,
        position: mx.array,
        k_layers: list,
        v_layers: list,
    ) -> tuple[mx.array, list, list]:
        """等长 batch 解码一步：所有行共享同一 offset，返回 logits 与更新后的 KV。

        与 eager 路径的区别：KV 数组（k_layers/v_layers）作为图输入显式传入、
        函数式返回新数组，不依赖 cache 对象，从而可被 ``mx.compile`` 整体编译。
        """
        x = self.weights.embed_tokens[token_ids]
        batch_size = token_ids.shape[0]
        capacity = k_layers[0].shape[2]
        # [B, 1, 1, capacity]：decode 单步只需限制 KV 的有效前缀 [0, position]
        attention_mask = (mx.arange(capacity) <= position).reshape(1, 1, 1, capacity)

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            qkv = self._linear(x_norm, layer.qkv_proj, layer.qkv_bias)
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
            q = mx.transpose(q.reshape(batch_size, 1, self.config.num_attention_heads, self.config.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            q = mx.fast.rope(q, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)
            k = mx.fast.rope(k, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)
            # slice_update 起点 [0, 0, position, 0]：写到当前 decode 位置
            start = mx.concatenate([
                mx.array([0, 0], dtype=mx.uint32),
                position.astype(mx.uint32).reshape(1),
                mx.array([0], dtype=mx.uint32),
            ])
            new_k_layers = list(k_layers)
            new_v_layers = list(v_layers)
            new_k_layers[layer_idx] = mx.slice_update(k_layers[layer_idx], k, start, axes=(0, 1, 2, 3))
            new_v_layers[layer_idx] = mx.slice_update(v_layers[layer_idx], v, start, axes=(0, 1, 2, 3))
            k_layers, v_layers = new_k_layers, new_v_layers
            attention = mx.fast.scaled_dot_product_attention(
                q, k_layers[layer_idx], v_layers[layer_idx], scale=self.scale, mask=attention_mask
            )
            attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(batch_size, 1, self.config.hidden_size)
            x = residual + self._linear(attention, layer.o_proj)
            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        logits = self._linear(self._rms_norm(x, self.weights.final_norm), self.weights.lm_head)
        return logits[:, -1, :], k_layers, v_layers

    def _static_decode_step_greedy(
        self,
        token_ids: mx.array,
        position: mx.array,
        k_layers: list,
        v_layers: list,
    ) -> tuple[mx.array, list, list]:
        """与 _static_decode_step 相同，但把 argmax 融合进编译图。

        直接返回 [batch] 的 token id 而非 [batch, vocab] logits，省去大
        logits 张量的物化，以及图外单独一次 argmax kernel 启动。
        """
        x = self.weights.embed_tokens[token_ids]
        batch_size = token_ids.shape[0]
        capacity = k_layers[0].shape[2]
        attention_mask = (mx.arange(capacity) <= position).reshape(1, 1, 1, capacity)

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            qkv = self._linear(x_norm, layer.qkv_proj, layer.qkv_bias)
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
            q = mx.transpose(q.reshape(batch_size, 1, self.config.num_attention_heads, self.config.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            q = mx.fast.rope(q, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)
            k = mx.fast.rope(k, self.config.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=position)
            # slice_update 起点 [0, 0, position, 0]：写到当前 decode 位置
            start = mx.concatenate([
                mx.array([0, 0], dtype=mx.uint32),
                position.astype(mx.uint32).reshape(1),
                mx.array([0], dtype=mx.uint32),
            ])
            new_k_layers = list(k_layers)
            new_v_layers = list(v_layers)
            new_k_layers[layer_idx] = mx.slice_update(k_layers[layer_idx], k, start, axes=(0, 1, 2, 3))
            new_v_layers[layer_idx] = mx.slice_update(v_layers[layer_idx], v, start, axes=(0, 1, 2, 3))
            k_layers, v_layers = new_k_layers, new_v_layers
            attention = mx.fast.scaled_dot_product_attention(
                q, k_layers[layer_idx], v_layers[layer_idx], scale=self.scale, mask=attention_mask
            )
            attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(batch_size, 1, self.config.hidden_size)
            x = residual + self._linear(attention, layer.o_proj)
            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        logits = self._linear(self._rms_norm(x, self.weights.final_norm), self.weights.lm_head)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        return next_tokens, k_layers, v_layers

    def get_compiled_static_decode_fn(
        self,
        batch_size: int,
        capacity: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """返回等长 batch（共享 offset）解码的已编译图，按固定形状缓存。

        首次调用时编译并缓存，之后命中直接复用；命中/未命中统计见
        ``compiled_static_decode_cache_stats``。
        """
        if batch_size <= 0 or capacity <= 0:
            raise ValueError("batch_size and capacity must be greater than 0")
        key = (self._quant_bits, batch_size, capacity, str(dtype))
        compiled = self._compiled_static_decodes.get(key)
        if compiled is None:
            self._compiled_static_decode_misses += 1
            compiled = mx.compile(self._static_decode_step)
            self._compiled_static_decodes[key] = compiled
        else:
            self._compiled_static_decode_hits += 1
        return compiled

    @property
    def compiled_static_decode_cache_stats(self) -> tuple[int, int]:
        """返回等长 batch 解码图的 (命中, 未命中) 次数。"""
        return self._compiled_static_decode_hits, self._compiled_static_decode_misses

    def get_compiled_static_decode_greedy_fn(
        self,
        batch_size: int,
        capacity: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """返回贪心（argmax 融合）版等长 batch 解码的已编译图。"""
        if batch_size <= 0 or capacity <= 0:
            raise ValueError("batch_size and capacity must be greater than 0")
        key = (self._quant_bits, batch_size, capacity, str(dtype))
        compiled = self._compiled_static_decode_greedy.get(key)
        if compiled is None:
            self._compiled_static_decode_greedy_misses += 1
            compiled = mx.compile(self._static_decode_step_greedy)
            self._compiled_static_decode_greedy[key] = compiled
        else:
            self._compiled_static_decode_greedy_hits += 1
        return compiled

    @property
    def compiled_static_decode_greedy_cache_stats(self) -> tuple[int, int]:
        """返回贪心版编译图的 (命中, 未命中) 次数。"""
        return self._compiled_static_decode_greedy_hits, self._compiled_static_decode_greedy_misses

    def _dynamic_decode_step(
        self,
        token_ids: mx.array,
        slot_offsets: mx.array,
        active_rows: mx.array,
        k_layers: list,
        v_layers: list,
    ) -> tuple[mx.array, list, list]:
        """纯张量、固定形状的动态解码图：各行游标独立。

        token_ids / slot_offsets / active_rows / KV 数组均为显式图输入，
        不依赖 Python 侧 cache 对象，因此可被 ``mx.compile`` 编译复用。
        """
        batch_size = token_ids.shape[0]
        capacity = k_layers[0].shape[2]
        x = self.weights.embed_tokens[token_ids]
        # 构造 [B, 1, 1, capacity] 的注意力掩码：活跃行只看到 [0, 自身游标]；
        # 非活跃行只留位置 0，保证数值合法（避免全掩码行产生 NaN），输出会被丢弃
        positions = mx.arange(capacity)[None, None, None, :]
        offsets = slot_offsets.astype(mx.int32)[:, None, None, None]
        valid_kv = positions <= offsets
        inactive_kv = mx.arange(capacity)[None, None, None, :] == 0
        active_mask = active_rows.astype(mx.bool_).reshape(batch_size, 1, 1, 1)
        attention_mask = mx.where(active_mask, valid_kv, inactive_kv)
        # 已释放的 slot 游标可能停在 capacity 上；钳制其（无用的）散射索引，
        # 保证固定形状图中每行的写入位置都在合法范围内
        write_offsets = mx.minimum(slot_offsets, mx.array(capacity - 1, dtype=slot_offsets.dtype))
        write_indices = mx.broadcast_to(
            write_offsets.astype(mx.uint32).reshape(batch_size, 1, 1, 1),
            (batch_size, k_layers[0].shape[1], 1, k_layers[0].shape[3]),
        )

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = x
            x_norm = self._rms_norm(x, layer.input_layernorm)
            qkv = self._linear(x_norm, layer.qkv_proj, layer.qkv_bias)
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
            q = mx.transpose(q.reshape(batch_size, 1, self.config.num_attention_heads, self.config.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim), (0, 2, 1, 3))
            # Per-row RoPE via tensor ops (no Python loop, compilable)
            positions = slot_offsets.astype(mx.float32)
            q = self._apply_rope_per_row(q, positions)
            k = self._apply_rope_per_row(k, positions)
            k_existing = mx.take_along_axis(k_layers[layer_idx], write_indices, axis=2)
            v_existing = mx.take_along_axis(v_layers[layer_idx], write_indices, axis=2)
            k_update = mx.where(active_mask, k, k_existing)
            v_update = mx.where(active_mask, v, v_existing)
            new_k_layers = list(k_layers)
            new_v_layers = list(v_layers)
            new_k_layers[layer_idx] = mx.put_along_axis(k_layers[layer_idx], write_indices, k_update, axis=2)
            new_v_layers[layer_idx] = mx.put_along_axis(v_layers[layer_idx], write_indices, v_update, axis=2)
            k_layers, v_layers = new_k_layers, new_v_layers
            attention = mx.fast.scaled_dot_product_attention(
                q, k_layers[layer_idx], v_layers[layer_idx], scale=self.scale, mask=attention_mask
            )
            attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(batch_size, 1, self.config.hidden_size)
            x = residual + self._linear(attention, layer.o_proj)
            residual = x
            x_norm = self._rms_norm(x, layer.post_attention_layernorm)
            gate_up = self._linear(x_norm, layer.gate_up_proj)
            gate, up = mx.split(gate_up, 2, axis=-1)
            x = residual + self._linear(nn.silu(gate) * up, layer.down_proj)

        logits = self._linear(self._rms_norm(x, self.weights.final_norm), self.weights.lm_head)
        return logits[:, -1, :], k_layers, v_layers

    def get_compiled_dynamic_decode_fn(
        self,
        batch_size: int,
        capacity: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """返回缓存的固定形状动态解码图。

        缓存键包含模型量化位数、batch 大小、上下文容量与 KV dtype。RoPE
        偏移以标量数组形式留在静态展开的图内，因此 batch/capacity 属于图
        形状的一部分——每种容量各缓存一份编译结果。
        """
        if batch_size <= 0 or capacity <= 0:
            raise ValueError("batch_size and capacity must be greater than 0")
        key = (self._quant_bits, batch_size, capacity, str(dtype))
        compiled = self._compiled_dynamic_decodes.get(key)
        if compiled is None:
            self._compiled_dynamic_decode_misses += 1
            compiled = mx.compile(self._dynamic_decode_step)
            self._compiled_dynamic_decodes[key] = compiled
        else:
            self._compiled_dynamic_decode_hits += 1
        return compiled

    @property
    def compiled_dynamic_decode_cache_stats(self) -> tuple[int, int]:
        """返回固定形状动态解码图的 (命中, 未命中) 次数。"""
        return self._compiled_dynamic_decode_hits, self._compiled_dynamic_decode_misses

    def _build_decode_attn_kernel(self, capacity: int) -> None:
        """构建 batch=1 GQA decode attention 的自定义 Metal kernel。

        使用在线（Flash）softmax + simd_sum 归约：每个 KV head 由一个
        32 线程的 SIMD 组负责，共享该 KV head 的所有 query head 复用同一份
        K/V 读取，KV 内存流量按 GQA 分组比倍数下降（如 7:1 分组则省 7 倍）。

        ``capacity`` 以编译期 ``#define`` 烧进 kernel，避免 fp16 精度损失；
        ``position`` 在运行期以两元素 fp16 数组（高/低位拆分）传入，
        支持超过 2048 的位置值。
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
        """运行自定义 Metal decode-attention kernel。

        Args:
            q: [num_heads, head_dim] query（已过 RoPE，压掉 batch 维）。
            k_cache_layer: [num_kv_heads, capacity, head_dim]。
            v_cache_layer: [num_kv_heads, capacity, head_dim]。
            position: 标量 mx.array — 当前 decode 偏移。
            capacity: KV cache 总容量。

        Returns:
            [num_heads, head_dim] 的注意力输出。
        """
        # 惰性构建 kernel（capacity 烧进 #define；容量变化时重建）
        if self._decode_attn_kernel is None or self._kernel_capacity != capacity:
            self._build_decode_attn_kernel(capacity)

        q_flat = q.reshape(-1)
        k_flat = k_cache_layer.reshape(-1)
        v_flat = v_cache_layer.reshape(-1)

        # 位置拆成高低两个 fp16 半段编码（pos = high*2048 + low），支持 > 2048
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
        """batch=1 单步贪心解码：返回 (next_token, 新 k_layers, 新 v_layers)。

        use_custom_attn=True 时注意力走自定义 Metal kernel，否则走
        ``mx.fast.scaled_dot_product_attention``。KV 数组显式进出、不依赖
        cache 对象，供 ``mx.compile`` 编译（见 get_compiled_decode_fn）。
        """
        x = self.weights.embed_tokens[token_ids]
        capacity = k_layers[0].shape[2]
        if not use_custom_attn:
            # decode 单步：mask 只需限制 KV 的有效前缀 [0, position]
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

            # slice_update 起点 [0, 0, position, 0]：写到当前 decode 位置
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
        """返回缓存好的 mx.compile 版 batch=1 解码步。

        编译函数只创建一次并跨调用复用；KV cache 数组作为参数传入
        （而非捕获的状态），因此同一张图适用于任意同形状的 cache。
        """
        if self._compiled_decode is None:
            self._compiled_decode = mx.compile(self._decode_greedy_step)
        return self._compiled_decode

    def forward(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """兼容辅助接口：只返回最后一个输入位置的 logits（内部走 prefill + 投影）。"""
        forward_hidden_output =  self.forward_hidden(token_ids, cache)
        output = self.project_last_hidden(forward_hidden_output)
        return output
