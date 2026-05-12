"""Batched Fresh Prefill 执行器

第四阶段核心组件：处理 past_key_values is None 的 prefill 请求，
将多个请求的 chunk padding 成 [B, max_chunk_len] 的统一 tensor，
一次 model forward 处理所有请求。

这是第四阶段里唯一真正 batched 的执行路径。

为什么 fresh prefill 能 batch：
- 请求没有历史 cache（past_key_values is None）
- 只需 padding + attention_mask 就可以一次 forward
- forward 后从每行最后一个有效位置取 logits 采样
- 把 batched past_key_values 拆回每个请求
"""

from __future__ import annotations

import torch

from miniservellm.runtime.attention_metadata import BatchedPrefillTensorBuilder
from miniservellm.runtime.outputs import StepEvent, StepEventType, TokenPayload, PrefillProgressPayload


class BatchedFreshPrefillExecutor:
    """Batched Fresh Prefill 执行器

    只处理 past_key_values is None 的请求。
    这是第四阶段里唯一真正 batch 的执行路径。

    Attributes:
        model_runner: 模型运行器
        sampler: 采样器
        tensor_builder: BatchedPrefillTensorBuilder
        request_index: request_id -> Request 映射
    """

    def __init__(self, model_runner, sampler, tensor_builder, request_index):
        self.model_runner = model_runner
        self.sampler = sampler
        self.tensor_builder = tensor_builder
        self.request_index = request_index

    @torch.no_grad()
    def run(self, requests, chunk_sizes: dict[str, int]) -> list[StepEvent]:
        """执行一批 fresh prefill 请求

        流程：
        1. 构建批量 tensor
        2. 模型前向传播
        3. 将批量输出拆解为每个请求的结果
        4. 更新每个请求的状态

        Args:
            requests: 本轮需要 fresh prefill 的请求列表
            chunk_sizes: request_id -> 本轮分配的 chunk token 数

        Returns:
            StepEvent 列表
        """
        if not requests:
            return []

        # 确保所有请求都没有 past_key_values
        for req in requests:
            assert req.past_key_values is None

        # 构建批量 tensor
        batch = self.tensor_builder.build(requests, chunk_sizes)

        # 一次 batched forward
        output = self.model_runner.forward_prefill(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            past_key_values=None,
        )

        logits = output.logits
        batched_past = output.past_key_values

        # 把 batched past_key_values 拆成每请求一份
        split_past_list = self._split_batched_past_key_values(
            batched_past_key_values=batched_past,
            batch_size=len(batch.request_order),
        )

        events: list[StepEvent] = []

        for i, req_id in enumerate(batch.request_order):
            req = self.request_index[req_id]
            chunk_len = batch.valid_lengths[i]

            # 推进 prefill 游标，保存拆分后的 past_key_values
            req.prefill_offset += chunk_len
            req.past_key_values = split_past_list[i]

            if req.is_prefill_done():
                # prefill 完成：从最后有效位置取 logits 采样首 token
                last_idx = batch.last_valid_indices[i]
                next_token_logits = logits[i, last_idx, :]
                next_token = self.sampler.sample(
                    next_token_logits,
                    temperature=req.sampling_params.temperature,
                    top_k=req.sampling_params.top_k,
                    top_p=req.sampling_params.top_p,
                )

                req.generated_token_ids.append(next_token)
                req.last_token_id = next_token
                req.status = "decoding"

                # 记录首 token 时间
                import time
                req.first_token_time = time.time()

                events.append(
                    StepEvent(
                        event_type=StepEventType.PREFILL_TO_DECODE,
                        request_id=req.request_id,
                        payload=TokenPayload(token_id=next_token),
                    )
                )
            else:
                # prefill 未完成，继续下一轮
                req.status = "prefilling"
                events.append(
                    StepEvent(
                        event_type=StepEventType.PREFILL_PROGRESS,
                        request_id=req.request_id,
                        payload=PrefillProgressPayload(
                            prefill_offset=req.prefill_offset,
                            prompt_len=req.prompt_len(),
                        ),
                    )
                )

        return events

    def _split_batched_past_key_values(self, batched_past_key_values, batch_size: int):
        """把 batched HF past_key_values 拆成每请求一份 cache

        HF 模型可能返回两种格式：
        - tuple: 经典格式，每层是 (key, value)，key/value shape [B, num_heads, seq_len, head_dim]
        - DynamicCache: 新格式，pv.layers[i].keys / pv.layers[i].values

        拆分时按 batch 维度切片，每个请求得到独立的一份 cache。

        Args:
            batched_past_key_values: 模型输出的 batched cache
            batch_size: batch 大小

        Returns:
            list of past_key_values，每个元素是一个请求的 cache
        """
        # 检测 cache 格式
        is_dynamic_cache = hasattr(batched_past_key_values, 'layers')

        if is_dynamic_cache:
            return self._split_dynamic_cache(batched_past_key_values, batch_size)
        else:
            return self._split_tuple_cache(batched_past_key_values, batch_size)

    def _split_tuple_cache(self, batched_past_key_values, batch_size: int):
        """拆分 tuple 格式的 past_key_values"""
        per_request = [[] for _ in range(batch_size)]

        for layer_past in batched_past_key_values:
            key, value = layer_past
            for i in range(batch_size):
                per_request[i].append(
                    (
                        key[i:i + 1].contiguous(),
                        value[i:i + 1].contiguous(),
                    )
                )

        return [tuple(x) for x in per_request]

    def _split_dynamic_cache(self, cache, batch_size: int):
        """拆分 DynamicCache 格式的 past_key_values

        DynamicCache 结构：cache.layers[i].keys / cache.layers[i].values
        每层 key/value shape: [B, num_heads, seq_len, head_dim]

        拆分后为每个请求创建一个新的 DynamicCache，
        其中每层的 keys/values 是原 cache 对应 batch index 的切片。
        """
        from transformers.cache_utils import DynamicCache

        per_request_caches: list[DynamicCache] = []
        for _ in range(batch_size):
            new_cache = DynamicCache()
            per_request_caches.append(new_cache)

        for layer_idx, layer in enumerate(cache.layers):
            keys = layer.keys    # [B, num_heads, seq_len, head_dim]
            values = layer.values

            for i in range(batch_size):
                # 按批次维度切片
                per_request_caches[i].update(
                    key_states=keys[i:i + 1].contiguous(),
                    value_states=values[i:i + 1].contiguous(),
                    layer_idx=layer_idx,
                )

        return per_request_caches
