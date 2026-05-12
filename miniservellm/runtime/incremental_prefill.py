"""Incremental Prefill 执行器

第四阶段组件：处理已经带有 past_key_values 的 prompt continuation。
由于不同请求的 cache 不同，HF 原生接口下无法 batch，所以逐请求执行。

为什么 incremental prefill 不能 batch：
- 每个请求有自己的 past_key_values（长度、内容都不同）
- HF 模型接口不支持把不同请求的 cache 拼成一个 batch
- 真正的 batched incremental prefill 需要统一 paged KV backend（第五阶段）
"""

from __future__ import annotations

import torch

from miniservellm.runtime.outputs import StepEvent, StepEventType, TokenPayload, PrefillProgressPayload


class IncrementalPrefillExecutor:
    """Incremental Prefill 执行器

    处理已经带有 past_key_values 的 prefill 请求，逐请求执行。

    Attributes:
        model_runner: 模型运行器
        sampler: 采样器
    """

    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    @torch.no_grad()
    def run_one(self, req, chunk_size: int) -> list[StepEvent]:
        """执行一个 incremental prefill 请求

        流程：
        1. 取出 [prefill_offset, prefill_offset + chunk_size) 的 chunk
        2. 携带 past_key_values forward
        3. 如果 prefill 完成，从最后位置取 logits 采样首 token

        Args:
            req: 当前请求
            chunk_size: 本轮分配的 chunk token 数

        Returns:
            StepEvent 列表
        """
        start = req.prefill_offset
        end = start + chunk_size
        chunk = req.prompt_token_ids[start:end]

        # 构造输入 [1, chunk_len]
        input_ids = torch.tensor([chunk], dtype=torch.long)

        # attention_mask 覆盖 past + 当前 chunk
        past_len = req.prefill_offset
        total_len = past_len + len(chunk)
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

        output = self.model_runner.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=req.past_key_values,
        )

        # 更新请求状态
        req.past_key_values = output.past_key_values
        req.prefill_offset += len(chunk)

        events: list[StepEvent] = []

        if req.is_prefill_done():
            # prefill 完成：采样首 token
            next_token_logits = output.logits[0, -1, :]
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
