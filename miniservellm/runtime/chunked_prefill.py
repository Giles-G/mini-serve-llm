"""Chunked Prefill 执行器

第三阶段核心组件：将长 prompt 分成多个 chunk 逐步 prefill，
而不是一次性处理完整个 prompt。

工作流程：
1. 每次只取 prompt 中 [prefill_cursor, prefill_cursor + chunk_size) 的 token
2. 将 chunk 送入模型 forward，更新 KV Cache
3. 如果 prefill 未完成，不产出 token，请求回到 prefilling_queue
4. 如果 prefill 完成，从最后一个位置的 logits 采样首 token，请求进入 decoding

优势：
- 长 prompt 不会独占 GPU，其他 decode 请求可以同时推进
- 与 decode-first 调度配合，已开始生成的请求保持流式输出
"""

from __future__ import annotations

import time
import torch

from miniservellm.runtime.outputs import ChunkedPrefillResult


class ChunkedPrefillExecutor:
    """Chunked Prefill 执行器

    Attributes:
        model_runner: 模型运行器，负责实际 forward
        sampler: 采样器，从 logits 中采样 token
        kv_cache_manager: KV Cache 管理器，保存 HF past_key_values
    """

    def __init__(self, model_runner, sampler, kv_cache_manager):
        self.model_runner = model_runner
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager

    def run_chunk(self, request, chunk_size: int | None = None):
        """执行一个 prefill chunk

        Args:
            request: 当前请求对象
            chunk_size: 调度器分配的本轮 token 预算，
                        如果为 None 则使用 request.chunk_size

        Returns:
            ChunkedPrefillResult 包含：
            - chunk_processed_tokens: 本轮处理的 token 数
            - produced_token: 产出的首 token id（prefill 未完成时为 None）
            - prefill_done: 是否完成全部 prefill
            - finished: 请求是否已完成（首 token 就满足终止条件时为 True）
        """
        # 已完成 prefill 的请求直接返回
        if request.prefill_done:
            return ChunkedPrefillResult(
                chunk_processed_tokens=0,
                produced_token=None,
                prefill_done=True,
                finished=False,
            )

        # 确定本轮实际 chunk 大小：取调度器分配值和 request 默认值的较小值
        effective_chunk_size = request.chunk_size
        if chunk_size is not None:
            effective_chunk_size = max(0, min(chunk_size, request.chunk_size))

        # 取出 [prefill_cursor, prefill_cursor + chunk_size) 范围的 token
        start = request.prefill_cursor
        end = min(start + effective_chunk_size, len(request.prompt_token_ids))
        chunk_token_ids = request.prompt_token_ids[start:end]

        # 没有可处理的 token，直接返回
        if len(chunk_token_ids) == 0:
            return ChunkedPrefillResult(
                chunk_processed_tokens=0,
                produced_token=None,
                prefill_done=request.prefill_done,
                finished=False,
            )

        # 构造输入 tensor [1, chunk_len]
        input_ids = torch.tensor([chunk_token_ids], dtype=torch.long)

        # attention_mask 必须覆盖 past_key_values 中的 token + 当前 chunk 的 token
        # 当有 past_key_values 时，HF 模型要求 attention_mask 长度 = past_len + current_len
        past_len = request.prefill_cursor  # 之前 chunk 已处理的 token 数
        total_len = past_len + len(chunk_token_ids)
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

        # 获取当前请求已有的 KV Cache（可能来自之前的 chunk）
        cache_handle = self.kv_cache_manager.get(request.request_id)
        past_key_values = cache_handle.past_key_values if cache_handle is not None else None

        # 执行 forward
        output = self.model_runner.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        # 更新 request 的 KV Cache 和 prefill 游标
        request.past_key_values = output.past_key_values
        request.advance_prefill_cursor(len(chunk_token_ids))

        # 更新 KVCacheManager，token_count = 已 prefill 的 token 数 + 已生成的 token 数
        current_total_tokens = request.prefill_cursor + len(request.generated_token_ids)
        self.kv_cache_manager.update(
            request.request_id,
            past_key_values=output.past_key_values,
            token_count=current_total_tokens,
        )

        # prefill 未完成：不产出 token，等下一轮继续
        if not request.prefill_done:
            return ChunkedPrefillResult(
                chunk_processed_tokens=len(chunk_token_ids),
                produced_token=None,
                prefill_done=False,
                finished=False,
            )

        # prefill 完成：从最后一个位置的 logits 采样首 token
        next_token_id = self.sampler.sample(
            output.logits[:, -1, :],
            temperature=request.sampling_params.temperature,
            top_k=request.sampling_params.top_k,
            top_p=request.sampling_params.top_p,
        )

        # 更新请求状态
        request.last_token_id = next_token_id
        request.generated_token_ids.append(next_token_id)

        # 记录首 token 时间
        if request.first_token_time is None:
            request.first_token_time = time.time()

        # 判断是否完成：达到最大长度或遇到 stop token
        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or next_token_id in request.sampling_params.stop_token_ids
        )

        # 更新 KVCacheManager 的 token_count
        self.kv_cache_manager.update(
            request.request_id,
            past_key_values=output.past_key_values,
            token_count=request.prefill_cursor + len(request.generated_token_ids),
        )

        if finished and request.finish_time is None:
            request.finish_time = time.time()

        return ChunkedPrefillResult(
            chunk_processed_tokens=len(chunk_token_ids),
            produced_token=next_token_id,
            prefill_done=True,
            finished=finished,
        )
