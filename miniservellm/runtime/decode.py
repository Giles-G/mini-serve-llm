"""Decode 执行器

负责自回归解码的每一步：输入上一个 token，配合 KV Cache 产出下一个 token。
第二阶段从 KVCacheManager 读取和更新 cache。
"""

import time
import torch


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
            (next_token_id, finished)
        """
        # 优先从 KVCacheManager 获取 cache，兼容 fallback 到 request.past_key_values
        cache_handle = self.kv_cache_manager.get(request.request_id)
        past_key_values = cache_handle.data if cache_handle is not None else request.past_key_values

        # 每步 decode 只输入上一个生成的 token
        input_ids = torch.tensor([[request.last_token_id]], dtype=torch.long)

        # 更严谨的 attention_mask：长度覆盖 prompt + 已生成 token。
        # HF past_key_values 已包含之前 token 的 KV，当前 input_ids 是最后一个生成 token。
        total_len = len(request.prompt_token_ids) + len(request.generated_token_ids)
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

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

        # 更新 cache manager，token_count = prompt + 已生成
        self.kv_cache_manager.update(
            request.request_id,
            past_key_values=output.past_key_values,
            token_count=len(request.prompt_token_ids) + len(request.generated_token_ids),
        )

        if finished:
            request.finish_time = time.time()

        return next_token_id, finished
