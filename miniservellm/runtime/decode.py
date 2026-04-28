"""Decode 执行器

负责自回归解码的每一步：输入上一个 token，配合 KV Cache 产出下一个 token。
每步只输入 1 个 token，利用 KV Cache 避免重复计算之前的注意力。
"""

import torch


class DecodeExecutor:
    """Decode 阶段执行器

    Attributes:
        model_runner: 模型运行器，负责实际的前向推理
        sampler: 采样器，从 logits 中采样 token
    """

    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    def step(self, request):
        """执行一步 decode：输入上一个 token，产出下一个 token

        步骤：
        1. 将上一个生成的 token 构造为 [1, 1] 输入
        2. 构造正确长度的 attention_mask（覆盖 prompt + 已生成 token）
        3. 调用模型 forward 得到 logits 和更新后的 KV Cache
        4. 采样下一个 token
        5. 判断是否达到终止条件

        Args:
            request: 请求对象，包含 past_key_values 和 sampling_params

        Returns:
            (next_token_id, finished) 元组
            - next_token_id: 本步采样的 token id
            - finished: 是否满足终止条件
        """
        # 只输入上一个生成的 token，形状 [1, 1]
        input_ids = torch.tensor([[request.last_token_id]], dtype=torch.long)

        # attention_mask 长度必须等于 past_key_values 的序列长度 + 当前 token
        # 否则模型无法正确计算注意力（这是 KV Cache 使用的关键）
        prompt_len = len(request.prompt_token_ids)
        generated_len = len(request.generated_token_ids)
        total_len = prompt_len + generated_len
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

        output = self.model_runner.forward_decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=request.past_key_values,
        )

        # 从 logits 采样下一个 token
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

        # 终止条件：达到最大生成长度 或 采样到 stop token
        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or next_token_id in request.sampling_params.stop_token_ids
        )

        return next_token_id, finished
