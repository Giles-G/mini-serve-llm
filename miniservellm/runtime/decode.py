"""Decode 执行器

第三阶段的 DecodeExecutor 适配 PagedKVCacheHandle，
从 KVCacheManager 获取 past_key_values 时使用 .past_key_values 而非 .data。
同时增加 sampling_params.stop_token_ids 的终止判断。
"""

from __future__ import annotations

import time
import torch

from miniservellm.runtime.outputs import DecodeStepResult


class DecodeExecutor:
    """Decode 阶段执行器

    Attributes:
        model_runner: 模型运行器，负责实际 forward
        sampler: 采样器，从 logits 中采样 token
        kv_cache_manager: KV Cache 管理器
    """

    def __init__(self, model_runner, sampler, kv_cache_manager):
        self.model_runner = model_runner
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager

    def step(self, request):
        """执行一步 decode

        Args:
            request: 当前请求对象

        Returns:
            DecodeStepResult
        """
        if request.last_token_id is None:
            raise ValueError(f"Request {request.request_id} has no last_token_id for decode.")

        # 从 KVCacheManager 获取 cache
        cache_handle = self.kv_cache_manager.get(request.request_id)
        past_key_values = cache_handle.past_key_values if cache_handle is not None else None

        # 每步 decode 只输入上一个生成的 token
        input_ids = torch.tensor([[request.last_token_id]], dtype=torch.long)

        # attention_mask 长度覆盖 prompt 已处理部分 + 已生成 token
        # HF past_key_values 已包含之前 token 的 KV，当前 input_ids 是最后一个生成 token
        attention_mask = torch.ones(1, request.total_sequence_length(), dtype=torch.long)

        output = self.model_runner.forward_decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        next_token_id = self.sampler.sample(
            output.logits[:, -1, :],
            temperature=request.sampling_params.temperature,
            top_k=request.sampling_params.top_k,
            top_p=request.sampling_params.top_p,
        )

        # 更新 request 状态
        request.past_key_values = output.past_key_values
        request.last_token_id = next_token_id
        request.generated_token_ids.append(next_token_id)

        if request.first_token_time is None:
            request.first_token_time = time.time()

        # 判断终止：达到最大长度或遇到 stop token
        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or next_token_id in request.sampling_params.stop_token_ids
        )

        # 更新 KVCacheManager，token_count = 已 prefill 的 token + 已生成 token
        self.kv_cache_manager.update(
            request.request_id,
            past_key_values=output.past_key_values,
            token_count=request.total_sequence_length(),
        )

        if finished and request.finish_time is None:
            request.finish_time = time.time()

        return DecodeStepResult(next_token_id=next_token_id, finished=finished)
