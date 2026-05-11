"""Prefill 执行器

负责将完整 prompt 一次性送入模型，初始化 KV Cache并产出第一个 token 。
第二阶段接入 KVCacheManager，由它统一管理 request_id -> past_key_values。
"""

import time
import torch

from miniservellm.runtime.outputs import PrefillResult


class PrefillExecutor:
    """Prefill 阶段执行器

    Attributes:
        model_runner: 模型运行器，负责实际 forward
        sampler: 采样器，从 logits 中采样 token
        kv_cache_manager: KV Cache 管理器，保存 HF past_key_values
    """

    def __init__(self, model_runner, sampler, kv_cache_manager):
        self.model_runner = model_runner
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager

    def run(self, request):
        """执行 prefill：输入完整 prompt，生成首 token

        Args:
            request: 当前请求对象

        Returns:
            PrefillResult: 首个生成的 token id
        """
        # prompt token ids -> [1, prompt_len]
        input_ids = torch.tensor([request.prompt_token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # 首次 forward 没有 past_key_values
        output = self.model_runner.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        # 只使用最后一个位置的 logits 采样下一个 token
        next_token_id = self.sampler.sample(
            output.logits[:, -1, :],
            temperature=request.sampling_params.temperature,
            top_k=request.sampling_params.top_k,
            top_p=request.sampling_params.top_p,
        )

        # 更新 request 运行状态
        request.past_key_values = output.past_key_values
        request.last_token_id = next_token_id
        request.generated_token_ids.append(next_token_id)
        request.prefill_done = True

        # 首 token 生成时间，用于 TTFT
        if request.first_token_time is None:
            request.first_token_time = time.time()

        # 将 KV Cache 写入 manager，token_count = prompt + 已生成
        self.kv_cache_manager.update(
            request.request_id,
            past_key_values=output.past_key_values,
            token_count=len(request.prompt_token_ids) + len(request.generated_token_ids),
        )

        return PrefillResult(next_token_id=next_token_id)
