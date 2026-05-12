"""调度器

第四阶段 Scheduler 直接从 RequestQueue 的 active_prefill / active_decode 中选择请求，
输出 SchedulePlan（decode_requests + prefill_requests + prefill_chunk_sizes）。

调度策略：decode-first + prefill token budget。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from miniservellm.scheduler.request import Request


@dataclass
class SchedulePlan:
    """调度计划

    Attributes:
        decode_requests: 本轮要执行 decode 的请求列表
        prefill_requests: 本轮要执行 prefill 的请求列表
        prefill_chunk_sizes: request_id -> 本轮分配的 chunk token 数
    """
    decode_requests: list[Request]
    prefill_requests: list[Request]
    prefill_chunk_sizes: dict[str, int] = field(default_factory=dict)


class Scheduler:
    """请求调度器

    策略：
    - decode-first：优先调度 decode 请求
    - prefill token budget：每步给 prefill 一个总预算，在请求间分配

    Attributes:
        max_decode_batch_size: 最大 decode batch 大小
        max_prefill_batch_size: 最大 prefill batch 大小
        prefill_token_budget: 每步 prefill 的总 token 预算
        max_prefill_chunk_size: 单个请求每步最大 prefill chunk 大小
    """

    def __init__(
        self,
        max_decode_batch_size: int = 8,
        max_prefill_batch_size: int = 8,
        prefill_token_budget: int = 256,
        max_prefill_chunk_size: int = 64,
    ):
        self.max_decode_batch_size = max_decode_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.prefill_token_budget = prefill_token_budget
        self.max_prefill_chunk_size = max_prefill_chunk_size

    def schedule(self, queue) -> SchedulePlan:
        """生成调度计划

        Args:
            queue: RequestQueue 实例

        Returns:
            SchedulePlan
        """
        # decode-first：优先取 decode 请求
        decode_requests = queue.active_decode[:self.max_decode_batch_size]

        # 在 token budget 内为 prefill 请求分配 chunk
        remaining_budget = self.prefill_token_budget
        prefill_requests: list[Request] = []
        prefill_chunk_sizes: dict[str, int] = {}

        candidates = queue.active_prefill[:self.max_prefill_batch_size]

        for req in candidates:
            if remaining_budget <= 0:
                break

            remaining = req.remaining_prompt_tokens()
            # chunk 大小 = min(剩余未处理, 最大chunk, 剩余预算)
            chunk = min(remaining, self.max_prefill_chunk_size, remaining_budget)

            if chunk <= 0:
                continue

            prefill_requests.append(req)
            prefill_chunk_sizes[req.request_id] = chunk
            remaining_budget -= chunk

        return SchedulePlan(
            decode_requests=decode_requests,
            prefill_requests=prefill_requests,
            prefill_chunk_sizes=prefill_chunk_sizes,
        )
