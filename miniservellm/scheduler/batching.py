"""Batching 策略

第二阶段先实现调度层面的 batching：Scheduler 每轮选出一批请求，
但执行层暂时仍然逐 request forward，后续可替换为真正的 tensor batch。
"""

from dataclasses import dataclass
from miniservellm.scheduler.request import Request


@dataclass
class BatchPlan:
    """一次 engine.step() 的执行计划

    Attributes:
        prefill_requests: 本轮要执行 prefill 的请求列表
        decode_requests: 本轮要执行 decode 的请求列表
    """
    prefill_requests: list[Request]
    decode_requests: list[Request]

    @property
    def total_size(self) -> int:
        """本轮总请求数"""
        return len(self.prefill_requests) + len(self.decode_requests)


class BatchingStrategy:
    """构造 BatchPlan 的策略

    当前实现支持 decode-first：优先调度 decode 请求，剩余 batch 空位再填 prefill。
    这符合 continuous batching 的常见策略，因为 decode 请求通常已经占用 KV Cache，
    优先推进可减少长尾延迟和 cache 占用时间。
    """

    def __init__(self, max_batch_size: int = 4, decode_first: bool = True):
        self.max_batch_size = max_batch_size
        self.decode_first = decode_first

    def build_batch(self, request_queue) -> BatchPlan:
        """从请求队列中构造一个 BatchPlan

        Args:
            request_queue: RequestQueue 实例

        Returns:
            当前 step 的 BatchPlan
        """
        prefill_requests = []
        decode_requests = []

        remaining = self.max_batch_size

        if self.decode_first:
            # decode 优先：先取已有 KV Cache 的请求继续推进
            decode_requests = request_queue.pop_decode_candidates(remaining)
            remaining -= len(decode_requests)

            # 如果 batch 还有空位，再补充新的 prefill 请求
            if remaining > 0:
                prefill_requests = request_queue.pop_prefill_candidates(remaining)
        else:
            # prefill 优先：用于实验不同调度策略
            prefill_requests = request_queue.pop_prefill_candidates(remaining)
            remaining -= len(prefill_requests)

            if remaining > 0:
                decode_requests = request_queue.pop_decode_candidates(remaining)

        return BatchPlan(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
        )
