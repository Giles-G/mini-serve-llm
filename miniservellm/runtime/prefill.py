"""Prefill 执行器

负责将完整 prompt 一次性送入模型，产出第一个 token 并初始化 KV Cache。
这是自回归生成的第一步，也叫做 "预填充" 阶段。
"""

import time
import torch


class PrefillExecutor:
    """Prefill 阶段执行器

    Attributes:
        model_runner: 模型运行器，负责实际的前向推理
        sampler: 采样器，从 logits 中采样 token
    """

    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    def run(self, request):
        """执行 prefill：输入完整 prompt，产出第一个 token

        步骤：
        1. 将 prompt token ids 构造为输入张量
        2. 调用模型 forward 得到 logits 和 KV Cache
        3. 从最后一个位置的 logits 采样第一个 token
        4. 将 KV Cache 和采样结果写回 request

        Args:
            request: 请求对象，包含 prompt_token_ids 和 sampling_params

        Returns:
            第一个生成的 token id
        """
        # 将 prompt token ids 构造为 [1, seq_len] 的输入张量
        input_ids = torch.tensor([request.prompt_token_ids], dtype=torch.long)

        # Prefill 阶段：无 KV Cache，attention_mask 全 1
        output = self.model_runner.forward_prefill(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=None,
        )

        # 从最后一个 token 位置的 logits 采样
        next_token_id = self.sampler.sample(
            output.logits[:, -1, :],
            temperature=request.sampling_params.temperature,
            top_k=request.sampling_params.top_k,
            top_p=request.sampling_params.top_p,
        )

        # 将 KV Cache 写回 request，供后续 decode 使用
        request.past_key_values = output.past_key_values
        request.last_token_id = next_token_id
        request.generated_token_ids.append(next_token_id)

        # 记录首 token 时间，用于计算 TTFT（Time To First Token）
        if request.first_token_time is None:
            request.first_token_time = time.time()

        return next_token_id
