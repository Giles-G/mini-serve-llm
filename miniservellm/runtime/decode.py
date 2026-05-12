"""Decode 执行器

第四阶段：decode 仍然逐请求执行，每个请求携带各自的 past_key_values。

为什么 decode 不能 batch：
- 每个请求有自己的 past_key_values（长度、内容都不同）
- HF 模型接口不支持把不同请求的 cache 拼成一个 batch
- 真正的 batched decode 需要统一 paged KV backend（第五阶段）
"""

from __future__ import annotations

import torch

from miniservellm.runtime.outputs import StepEvent, StepEventType, TokenPayload, FinishedPayload


class DecodeExecutor:
    """Decode 执行器

    逐请求执行 decode，每个请求携带各自的 past_key_values。

    Attributes:
        model_runner: 模型运行器
        sampler: 采样器
    """

    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    @torch.no_grad()
    def run_one(self, req) -> list[StepEvent]:
        """执行一个 decode 请求

        Args:
            req: 当前请求

        Returns:
            StepEvent 列表
        """
        assert req.last_token_id is not None

        # 输入上一个生成的 token [1, 1]
        input_ids = torch.tensor([[req.last_token_id]], dtype=torch.long)

        # attention_mask 覆盖完整序列
        attention_mask = torch.ones(1, req.total_sequence_length(), dtype=torch.long)

        output = self.model_runner.forward_decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=req.past_key_values,
        )

        # 更新请求状态
        req.past_key_values = output.past_key_values

        next_token_logits = output.logits[0, -1, :]
        next_token = self.sampler.sample(
            next_token_logits,
            temperature=req.sampling_params.temperature,
            top_k=req.sampling_params.top_k,
            top_p=req.sampling_params.top_p,
        )

        req.generated_token_ids.append(next_token)
        req.last_token_id = next_token

        events: list[StepEvent] = [
            StepEvent(
                event_type=StepEventType.DECODE_TOKEN,
                request_id=req.request_id,
                payload=TokenPayload(token_id=next_token),
            )
        ]

        # 判断终止
        finished = (
            len(req.generated_token_ids) >= req.sampling_params.max_new_tokens
            or next_token in req.sampling_params.stop_token_ids
        )

        if finished:
            req.status = "finished"
            if req.finish_time is None:
                import time
                req.finish_time = time.time()
            events.append(
                StepEvent(
                    event_type=StepEventType.FINISHED,
                    request_id=req.request_id,
                    payload=FinishedPayload(reason="eos_or_max_new_tokens"),
                )
            )

        return events
