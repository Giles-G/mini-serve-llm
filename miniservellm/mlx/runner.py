"""Batch=1 MLX Qwen2 greedy inference runner.

MLX 后端（Apple Silicon）的 Qwen2 推理 Runner。

职责划分：
- 本 Runner 只做模型侧操作（prefill / decode / 贪心解码循环），
  请求生命周期与采样策略由上层 serving engine 负责；
- 提供「等长静态 batch」与「不等长动态 batch / 槽位（slot）」两类原语，
  前者用于基准测试，后者为 phase-2 正确性优先的动态 batch 路径；
- 不使用 Paged KV 或 Scheduler，属于独立 FP16 跑通路径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlx.core as mx

from miniservellm.config import ModelConfig
from miniservellm.mlx.adapter import load_qwen2_weights
from miniservellm.mlx.qwen2 import MLXKVCache, MLXQwen2


@dataclass
class MLXGenerationResult:
    """单条请求的生成结果与分阶段计时。"""

    generated_token_ids: List[int]  # 包含 prefill 产出的首个 token
    prefill_seconds: float
    decode_seconds: float

    @property
    def decode_tok_s(self) -> float:
        """decode 阶段吞吐；首个 token 由 prefill 产出，需从计数中减去。"""
        decode_tokens = max(0, len(self.generated_token_ids) - 1)
        return decode_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0


class MLXQwen2Runner:
    """Independent Apple Silicon FP16 runner; does not use Paged KV or Scheduler."""

    def __init__(self, model_config: ModelConfig, safetensors_path: str | Path):
        self.model_config = model_config
        self.weights = load_qwen2_weights(safetensors_path, model_config)
        self.model = MLXQwen2(model_config, self.weights)
        # 强制 eager 地加载 embedding 权重到 GPU，避免首次 forward 时才搬运权重
        mx.eval(self.weights.embed_tokens)

    def prefill_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
    ) -> tuple[MLXKVCache, mx.array]:
        """Prefill an equal-length request batch and return cache plus [B, vocab] logits.

        The serving engine owns request lifecycle and sampling; this primitive only
        performs the model-side operation needed to admit one fixed batch.

        等长 batch 的 prefill 原语：
        - 要求所有 prompt token 长度一致（静态 batch，无需 padding/mask）；
        - KV cache 容量 = prompt 长度 + max_new_tokens，一次分配到位；
        - 返回最后一个位置的 logits，供上层做采样/argmax 后继续 decode。
        """
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        lengths = {len(prompt) for prompt in prompt_token_ids}
        if len(lengths) != 1:
            raise ValueError("prefill_batch requires prompts with equal token length")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        batch_size = len(prompt_token_ids)
        capacity = next(iter(lengths)) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        prompt = mx.array(prompt_token_ids, dtype=mx.uint32)
        # 整段 prompt 一次 forward；只取最后一个位置的 logits
        logits = self.model.forward(prompt, cache)[:, -1, :]
        return cache, logits

    def decode_batch(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """Decode one token for every row of an existing fixed MLX batch.

        静态 batch 的单步解码：每行喂入 1 个 token，共享同一个 KV 写入位置
        （cache.offset），返回 [batch, vocab] 的最后位置 logits。
        """
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        return self.model.forward(token_ids.reshape(cache.batch_size, 1), cache)[:, -1, :]

    def prefill_dynamic_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
    ) -> tuple[MLXKVCache, mx.array]:
        """Prefill unequal prompts into one per-slot-offset cache.

        Each row is prefetched independently to preserve its true causal length,
        then copied into its row of a shared batch cache. Decode thereafter is
        batched. This is the correctness-first phase-2 path; phase 5 will replace
        these row copies with masked padded prefill and batched KV scatter.

        不等长 prompt 的动态 prefill（phase-2 正确性优先路径）：
        - 每行先独立 prefill 到自己的临时 row_cache，保留真实因果长度；
        - 再逐层把 row_cache 的 K/V 拷贝进共享 batch cache 的对应行（slot）；
        - 记录每行的 slot_offsets（真实长度），供动态 decode 使用；
        - 之后 decode 即可整 batch 并行。
        """
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        if any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt_token_ids must not contain empty prompts")

        batch_size = len(prompt_token_ids)
        # 容量按最长 prompt 计（短行只占用前缀，其余为空槽）
        capacity = max(len(prompt) for prompt in prompt_token_ids) + max_new_tokens
        batch_cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        logits_rows: List[mx.array] = []
        for row, prompt_ids in enumerate(prompt_token_ids):
            # 每行独立 prefill，避免 padding 影响注意力
            row_cache = MLXKVCache(self.model_config, capacity)
            logits = self.model.forward(mx.array([prompt_ids], dtype=mx.uint32), row_cache)[:, -1, :]
            logits_rows.append(logits)
            # 逐层把该行的 K/V 拷入 batch cache 的第 row 行（从 [row,0,0,0] 起）
            for layer_idx in range(self.model_config.num_hidden_layers):
                start = mx.array([row, 0, 0, 0], dtype=mx.uint32)
                batch_cache.k_layers[layer_idx] = mx.slice_update(
                    batch_cache.k_layers[layer_idx], row_cache.k_layers[layer_idx], start, axes=(0, 1, 2, 3)
                )
                batch_cache.v_layers[layer_idx] = mx.slice_update(
                    batch_cache.v_layers[layer_idx], row_cache.v_layers[layer_idx], start, axes=(0, 1, 2, 3)
                )
        # 记录每行真实 KV 长度，动态 decode 时各按自己的 offset 写入/读取
        batch_cache.set_slot_offsets([len(prompt) for prompt in prompt_token_ids])
        return batch_cache, mx.concatenate(logits_rows, axis=0)

    def prefill_dynamic_slots(
        self,
        cache: MLXKVCache,
        slot_indices: List[int],
        prompt_token_ids: List[List[int]],
    ) -> mx.array:
        """Prefill requests independently and install their state in existing slots.

        The shared cache keeps a fixed batch shape. Only the supplied rows are
        replaced, which lets the serving engine refill a completed slot without
        disturbing other requests that are still decoding.

        槽位式动态 prefill：把新请求的 KV 状态安装到共享 cache 的指定槽位。
        只替换传入的行，其余行不受影响 —— 引擎可趁某请求结束后原位补充新请求，
        不打断仍在解码的其他请求。
        """
        if len(slot_indices) != len(prompt_token_ids) or not slot_indices:
            raise ValueError("slot_indices and prompt_token_ids must be non-empty and equally sized")
        if len(set(slot_indices)) != len(slot_indices):
            raise ValueError("slot_indices must be unique")
        if any(slot < 0 or slot >= cache.batch_size for slot in slot_indices):
            raise ValueError("slot index is outside cache batch size")
        if any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt_token_ids must not contain empty prompts")
        # prompt 长度必须小于 cache.max_context，给后续 decode 留出写入空间
        if any(len(prompt) >= cache.max_context for prompt in prompt_token_ids):
            raise ValueError("prompt leaves no decode capacity in the shared cache")

        logits_rows: List[mx.array] = []
        for slot, prompt_ids in zip(slot_indices, prompt_token_ids):
            row_cache = MLXKVCache(self.model_config, cache.max_context)
            logits = self.model.forward(
                mx.array([prompt_ids], dtype=mx.uint32), row_cache
            )[:, -1, :]
            logits_rows.append(logits)
            # 逐层把该请求的 K/V 拷入共享 cache 的 slot 行
            for layer_idx in range(self.model_config.num_hidden_layers):
                start = mx.array([slot, 0, 0, 0], dtype=mx.uint32)
                cache.k_layers[layer_idx] = mx.slice_update(
                    cache.k_layers[layer_idx],
                    row_cache.k_layers[layer_idx],
                    start,
                    axes=(0, 1, 2, 3),
                )
                cache.v_layers[layer_idx] = mx.slice_update(
                    cache.v_layers[layer_idx],
                    row_cache.v_layers[layer_idx],
                    start,
                    axes=(0, 1, 2, 3),
                )
            cache.slot_offsets[slot] = len(prompt_ids)
        return mx.concatenate(logits_rows, axis=0)

    def decode_static_batch(
        self,
        token_ids: mx.array,
        cache: MLXKVCache,
        compiled_decode: bool = False,
        greedy: bool = False,
    ) -> mx.array:
        """Decode a full batch sharing one KV cursor without dynamic masking.

        When ``greedy=True`` and ``compiled_decode=True``, uses an argmax-fused
        compiled graph that returns ``[batch]`` token ids instead of
        ``[batch, vocab]`` logits, avoiding logits materialisation.

        静态 batch 单步解码：所有行共享同一个 KV 写入游标（cache.offset），
        无需动态掩码。compiled_decode=True 时走 mx.compile 编译图，
        其中 greedy 版本把 argmax 融合进编译图，直接返回 [batch] token id，
        避免物化 [batch, vocab] 的 logits 张量。
        """
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        # 静态 batch 要求所有行长度一致（slot_offsets 全相同）
        if len(set(cache.slot_offsets)) != 1:
            raise ValueError("decode_static_batch requires equal slot offsets")
        if compiled_decode:
            if greedy:
                # argmax 融合的编译解码：返回 [batch] token id
                compiled_step = self.model.get_compiled_static_decode_greedy_fn(
                    cache.batch_size, cache.max_context, cache.k_layers[0].dtype
                )
                next_tokens, cache.k_layers, cache.v_layers = compiled_step(
                    token_ids.reshape(cache.batch_size, 1),
                    mx.array(cache.offset, dtype=mx.int32),
                    cache.k_layers,
                    cache.v_layers,
                )
                cache.advance(1)
                return next_tokens
            # 普通编译解码：返回 [batch, vocab] logits
            compiled_step = self.model.get_compiled_static_decode_fn(
                cache.batch_size, cache.max_context, cache.k_layers[0].dtype
            )
            logits, cache.k_layers, cache.v_layers = compiled_step(
                token_ids.reshape(cache.batch_size, 1),
                mx.array(cache.offset, dtype=mx.int32),
                cache.k_layers,
                cache.v_layers,
            )
            cache.advance(1)
            return logits
        # 未开编译时退回 eager 路径
        return self.decode_batch(token_ids, cache)

    def decode_dynamic_batch(
        self,
        token_ids: mx.array,
        cache: MLXKVCache,
        active_rows: List[bool],
        compiled_decode: bool = False,
    ) -> mx.array:
        """Decode one token using each active row's independent KV offset.

        动态 batch 单步解码：各行按各自的 slot_offsets 独立读写 KV；
        active_rows 标记本步参与解码的行（结束/空槽的行被掩码跳过，
        但 batch 形状保持不变）。compiled_decode=True 时走编译版本。
        """
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        if compiled_decode:
            compiled_step = self.model.get_compiled_dynamic_decode_fn(
                cache.batch_size, cache.max_context, cache.k_layers[0].dtype
            )
            logits, cache.k_layers, cache.v_layers = compiled_step(
                token_ids.reshape(cache.batch_size, 1),
                mx.array(cache.slot_offsets, dtype=mx.uint32),
                mx.array(active_rows, dtype=mx.bool_),
                cache.k_layers,
                cache.v_layers,
            )
            # 只有 active 的行推进各自的 offset
            cache.advance_slots(active_rows)
            return logits
        # eager 动态路径：先取最后一个有效位置的 hidden，再投影到词表
        hidden = self.model.forward_hidden_dynamic_decode(
            token_ids.reshape(cache.batch_size, 1), cache, active_rows
        )
        return self.model.project_last_hidden(hidden)[:, -1, :]

    def generate_greedy_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        disable_eos: bool = False,
        compiled_decode: bool = False,
        use_custom_attn: bool = False,
    ) -> List[MLXGenerationResult]:
        """Same-length batch greedy baseline for MLX throughput experiments.

        等长 batch 的贪心解码基线（吞吐实验用）：
        - prefill：整段 prompt 一次 forward，argmax 得到首个 token；
        - decode：循环单步贪心，三种路径二选一 ——
            use_custom_attn  > 自定义 Metal kernel（仅 batch=1）
            compiled_decode  > mx.compile 编译解码（仅 batch=1）
            其他             > eager forward + argmax
        - disable_eos=True 时固定生成 max_new_tokens 步（基准测试口径）。
        """
        if not prompt_token_ids:
            return []
        lengths = {len(prompt) for prompt in prompt_token_ids}
        if len(lengths) != 1:
            raise ValueError("generate_greedy_batch requires prompts with equal token length")
        if max_new_tokens <= 0:
            return [MLXGenerationResult([], 0.0, 0.0) for _ in prompt_token_ids]

        batch_size = len(prompt_token_ids)
        # 编译/自定义 kernel 路径当前只实现 batch=1；两者互斥
        if (compiled_decode or use_custom_attn) and batch_size != 1:
            raise ValueError("compiled_decode/use_custom_attn currently supports batch_size=1 only")
        if compiled_decode and use_custom_attn:
            raise ValueError("compiled_decode and use_custom_attn are mutually exclusive")
        capacity = next(iter(lengths)) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        prompt = mx.array(prompt_token_ids, dtype=mx.uint32)

        # ---------- prefill：整段一次 forward，取最后位置 argmax ----------
        prefill_start = time.perf_counter()
        logits = self.model.forward(prompt, cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tokens)  # 强制执行，prefill 计时到此为止
        prefill_seconds = time.perf_counter() - prefill_start

        compiled_step = None
        if compiled_decode:
            compiled_step = self.model.get_compiled_decode_fn()

        # 首个 token 来自 prefill，先写入生成序列
        generated = [[int(token)] for token in next_tokens.tolist()]
        decode_start = time.perf_counter()
        while len(generated[0]) < max_new_tokens:
            # 整 batch 全部以 EOS 收尾时提前结束（disable_eos 时不判）
            if not disable_eos and eos_token_id is not None:
                if all(tokens[-1] == eos_token_id for tokens in generated):
                    break
            if use_custom_attn:
                # 自定义 Metal kernel 路径：直接产出 token id 与更新后的 KV
                next_tokens, cache.k_layers, cache.v_layers = self.model._decode_greedy_step(
                    next_tokens.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers, use_custom_attn=True
                )
                cache.advance(1)
            elif compiled_decode:
                # mx.compile 编译路径：argmax 已融合进编译图
                assert compiled_step is not None
                next_tokens, cache.k_layers, cache.v_layers = compiled_step(
                    next_tokens.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers
                )
                cache.advance(1)
            else:
                # eager 路径：普通 forward + argmax
                logits = self.model.forward(next_tokens.reshape(batch_size, 1), cache)
                next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_tokens)  # 每步强制执行，构成同步点
            # 已产出 EOS 的行停止追加（但继续参与 batch 解码，保持形状不变）
            for index, token in enumerate(next_tokens.tolist()):
                if disable_eos or eos_token_id is None or generated[index][-1] != eos_token_id:
                    generated[index].append(int(token))
        decode_seconds = time.perf_counter() - decode_start
        return [
            MLXGenerationResult(tokens, prefill_seconds, decode_seconds)
            for tokens in generated
        ]

    def generate_greedy(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        disable_eos: bool = False,
        compiled_decode: bool = False,
        use_custom_attn: bool = False,
    ) -> MLXGenerationResult:
        """单条请求的贪心解码（batch=1），逻辑与 generate_greedy_batch 相同，
        仅少了 batch 维度，供单请求基准与功能验证使用。"""
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if compiled_decode and use_custom_attn:
            raise ValueError("compiled_decode and use_custom_attn are mutually exclusive")
        if max_new_tokens <= 0:
            return MLXGenerationResult([], 0.0, 0.0)

        capacity = len(prompt_token_ids) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity)
        prompt = mx.array([prompt_token_ids], dtype=mx.uint32)

        # ---------- prefill ----------
        prefill_start = time.perf_counter()
        logits = self.model.forward(prompt, cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        prefill_seconds = time.perf_counter() - prefill_start

        compiled_step = None
        if compiled_decode:
            compiled_step = self.model.get_compiled_decode_fn()

        generated = [int(next_token.item())]
        decode_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            # 上一步生成 EOS 即停止
            if not disable_eos and eos_token_id is not None and generated[-1] == eos_token_id:
                break
            if use_custom_attn:
                # 自定义 Metal kernel 路径
                next_token, cache.k_layers, cache.v_layers = self.model._decode_greedy_step(
                    next_token.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers, use_custom_attn=True
                )
                cache.advance(1)
            elif compiled_decode:
                # mx.compile 编译路径
                assert compiled_step is not None
                next_token, cache.k_layers, cache.v_layers = compiled_step(
                    next_token.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers
                )
                cache.advance(1)
            else:
                # eager 路径
                logits = self.model.forward(next_token.reshape(1, 1), cache)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            generated.append(int(next_token.item()))
        decode_seconds = time.perf_counter() - decode_start

        return MLXGenerationResult(generated, prefill_seconds, decode_seconds)
